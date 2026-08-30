"""Torch rungs: deepfm, dcnv2, mmoe, ple, cwm, din_lite. CPU-first; CUDA/MPS
auto-detected, never required. Deterministic for a fixed seed on CPU."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from recsys.losses import LOSSES, bce
from recsys.models.base import Recommender

AUX_TASKS = ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate"]


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mlp(d_in, hidden, d_out):
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class _Trunk(nn.Module):
    """Embeds every categorical column (+1 shift so -1 fills hit pad slot 0),
    standardizes the dense columns; forward -> (embeds (B, n_cat, dim), dense (B, n_dense))."""

    def __init__(self, meta, X_train, dim=16, exclude_prefix=("sq_",)):
        super().__init__()
        cols = meta["columns"]
        self.cat_idx = [i for i in meta["categorical_idx"]
                        if not cols[i].startswith(exclude_prefix)]
        self.dense_idx = [i for i in range(len(cols))
                          if i not in set(meta["categorical_idx"])
                          and not cols[i].startswith(exclude_prefix)]
        self.sizes = []
        for i in self.cat_idx:
            fd = meta["field_dims"].get(cols[i])
            self.sizes.append((fd if fd else int(X_train[:, i].max()) + 2) + 1)
        self.emb = nn.ModuleList(nn.Embedding(s, dim) for s in self.sizes)
        for e in self.emb:
            nn.init.normal_(e.weight, 0, 0.01)
        mu = X_train[:, self.dense_idx].mean(0)
        sd = X_train[:, self.dense_idx].std(0) + 1e-6
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
        self.n_cat, self.dim, self.d_dense = len(self.cat_idx), dim, len(self.dense_idx)
        self.d_out = self.n_cat * dim + self.d_dense
        self._cols = cols

    def forward(self, X):
        cats = (X[:, self.cat_idx].long() + 1)
        embs = torch.stack(
            [e(cats[:, i].clamp(0, s - 1)) for i, (e, s) in enumerate(zip(self.emb, self.sizes))],
            dim=1)
        dense = ((X[:, self.dense_idx] - self.mu) / self.sd).clamp(-10, 10)
        return embs, dense

    def flat(self, X):
        embs, dense = self(X)
        return torch.cat([embs.flatten(1), dense], dim=1)


class _TorchRec(Recommender):
    defaults = {"dim": 16, "lr": 1e-3, "batch": 4096, "epochs": 30, "patience": 2,
                "subsample": 500_000, "loss": "bce", "weight_decay": 1e-6}

    def _build(self, meta, X_train):
        raise NotImplementedError

    def _forward(self, X):
        raise NotImplementedError

    def _loss(self, out, yb, seg, auxb):
        return LOSSES[self.cfg["loss"]](out, yb, seg)

    def fit(self, X_train, y_train, groups_train, aux_train=None,
            X_val=None, y_val=None, groups_val=None, time_budget=300, seed=0):
        self.cfg = {**self.defaults, **self.config}
        cfg = self.cfg
        torch.manual_seed(seed)
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
        self.device = _device()
        rng = np.random.default_rng(seed)
        n = len(y_train)
        if cfg["subsample"] and n > cfg["subsample"]:
            keep = rng.choice(n, cfg["subsample"], replace=False)
            self.info["subsampled_rows"] = int(cfg["subsample"])
        else:
            keep = np.arange(n)
        order = keep[np.argsort(groups_train[keep], kind="stable")]
        Xs = torch.tensor(X_train[order], dtype=torch.float32)
        ys = torch.tensor(y_train[order], dtype=torch.float32)
        auxs = {k: torch.tensor(v[order], dtype=torch.float32)
                for k, v in (aux_train or {}).items()}
        gs = groups_train[order]
        starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
        bounds = np.r_[starts, len(gs)]
        self._build(self.meta, X_train[order])
        self.net.to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"],
                               weight_decay=cfg["weight_decay"])
        deadline = self._deadline(time_budget)
        fixed = cfg.get("rounds")
        best, best_state, bad, used = -1.0, None, 0, 0
        snapshots = []  # (primary, state_dict) per epoch, for snapshot averaging
        n_users = len(starts)
        for ep in range(1, cfg["epochs"] + 1):
            self.net.train()
            uorder = rng.permutation(n_users)
            row_batches, cur, size = [], [], 0
            for u in uorder:
                cur.append(u)
                size += bounds[u + 1] - bounds[u]
                if size >= cfg["batch"]:
                    row_batches.append(cur)
                    cur, size = [], 0
            if cur:
                row_batches.append(cur)
            for users in row_batches:
                idx = np.concatenate([np.arange(bounds[u], bounds[u + 1]) for u in users])
                seg = np.repeat(np.arange(len(users)), [bounds[u + 1] - bounds[u] for u in users])
                Xb = Xs[idx].to(self.device)
                yb = ys[idx].to(self.device)
                segb = torch.tensor(seg, dtype=torch.int64, device=self.device)
                auxb = {k: v[idx].to(self.device) for k, v in auxs.items()}
                loss = self._loss(self._forward(Xb), yb, segb, auxb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if time.time() > deadline:
                    break
            used = ep
            if fixed or X_val is None:
                if (fixed and ep >= fixed) or time.time() > deadline:
                    break
                continue
            primary = self._val_primary(self.predict(X_val, groups_val))
            if cfg.get("snapshot_k"):
                snapshots.append((primary, {k: v.detach().clone()
                                            for k, v in self.net.state_dict().items()}))
            if primary > best + 1e-5:
                best, bad = primary, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
            if bad >= cfg["patience"] or time.time() > deadline:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        k = int(cfg.get("snapshot_k") or 0)
        if k > 1 and len(snapshots) > 1:
            # snapshot ensemble: average predictions of the top-k epoch checkpoints,
            # selected on validation (same information early stopping already uses)
            snapshots.sort(key=lambda s: -s[0])
            self._snapshots = [s[1] for s in snapshots[:k]]
            ens = self._val_primary(self.predict(X_val, groups_val))
            if ens > best:
                best = ens
                self.info["snapshot_ensemble"] = {"k": len(self._snapshots), "primary": ens}
            else:
                self._snapshots = None  # single best state wins
        self.info.update({"rounds_used": used,
                          "best_val_primary": best if best > 0 else None,
                          "config": {k2: v for k2, v in cfg.items()}})
        self._post_fit(X_val, y_val, groups_val)
        return self

    def _post_fit(self, X_val, y_val, groups_val):
        pass

    @torch.no_grad()
    def predict(self, X, groups):
        states = getattr(self, "_snapshots", None)
        if states:
            current = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            preds = []
            for sd in states:
                self.net.load_state_dict(sd)
                preds.append(self._predict_once(X))
            self.net.load_state_dict(current)
            return np.mean(preds, axis=0).astype(np.float32)
        return self._predict_once(X)

    @torch.no_grad()
    def _predict_once(self, X):
        self.net.eval()
        out = []
        for i in range(0, len(X), 65_536):
            Xb = torch.tensor(X[i:i + 65_536], dtype=torch.float32, device=self.device)
            out.append(self._score(Xb).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def _score(self, Xb):
        return self._forward(Xb)


class DeepFM(_TorchRec):
    def _build(self, meta, X_train):
        trunk = _Trunk(meta, X_train, dim=self.cfg["dim"])
        lin = nn.ModuleList(nn.Embedding(s, 1) for s in trunk.sizes)
        for e in lin:
            nn.init.zeros_(e.weight)
        self.net = nn.ModuleDict({
            "trunk": trunk, "lin": lin,
            "lin_dense": nn.Linear(trunk.d_dense, 1),
            "mlp": _mlp(trunk.d_out, [128, 64], 1),
        })

    def _forward(self, X):
        trunk = self.net["trunk"]
        embs, dense = trunk(X)
        cats = (X[:, trunk.cat_idx].long() + 1)
        first = sum(self.net["lin"][i](cats[:, i].clamp(0, s - 1)).squeeze(-1)
                    for i, s in enumerate(trunk.sizes)) + self.net["lin_dense"](dense).squeeze(-1)
        s = embs.sum(1)
        fm = 0.5 * ((s ** 2).sum(1) - (embs ** 2).sum(dim=(1, 2)))
        deep = self.net["mlp"](torch.cat([embs.flatten(1), dense], 1)).squeeze(-1)
        return first + fm + deep


class _CrossNet(nn.Module):
    def __init__(self, d, layers=3):
        super().__init__()
        self.ws = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))

    def forward(self, x0):
        x = x0
        for w in self.ws:
            x = x0 * w(x) + x
        return x


class DCNv2(_TorchRec):
    def _build(self, meta, X_train):
        trunk = _Trunk(meta, X_train, dim=self.cfg["dim"])
        self.net = nn.ModuleDict({
            "trunk": trunk,
            "cross": _CrossNet(trunk.d_out),
            "mlp": _mlp(trunk.d_out, [128], 64),
            "head": nn.Linear(trunk.d_out + 64, 1),
        })

    def _forward(self, X):
        flat = self.net["trunk"].flat(X)
        return self.net["head"](torch.cat([self.net["cross"](flat),
                                           self.net["mlp"](flat)], 1)).squeeze(-1)


class _MultiTask(_TorchRec):
    """Shared plumbing for mmoe/ple: long_view + aux binary heads + watch-ratio head,
    optional grid-searched head combination chosen on validation. Config knobs:
    aux_tasks (list), aux_weight, wr_weight."""

    @property
    def tasks(self):
        return ["long_view"] + self.cfg.get("aux_tasks", AUX_TASKS) + ["watch_ratio"]

    def _loss(self, outs, yb, seg, auxb):
        loss = LOSSES[self.cfg.get("main_loss", "bce")](outs[:, 0], yb, seg)
        aw = self.cfg.get("aux_weight", 0.3)
        for i, t in enumerate(self.cfg.get("aux_tasks", AUX_TASKS), start=1):
            loss = loss + aw * bce(outs[:, i], auxb[t])
        wr = (auxb["play_time_ms"] / auxb["duration_ms"].clamp(1)).clamp(0, 2)
        loss = loss + self.cfg.get("wr_weight", 0.5) * nn.functional.huber_loss(outs[:, -1], wr)
        return loss

    def _score(self, Xb):
        outs = self._forward(Xb)
        w = getattr(self, "head_weights", None)
        if w:
            return outs[:, 0] + w["click"] * outs[:, 1] + w["wr"] * outs[:, -1]
        return outs[:, 0]

    def _post_fit(self, X_val, y_val, groups_val):
        if X_val is None or not self.cfg.get("head_grid", True):
            return
        best = (self._val_primary(self.predict(X_val, groups_val)), None)
        for c in (-0.25, 0.0, 0.25, 0.5):
            for w in (0.0, 0.25, 0.5, 0.75):
                if c == w == 0.0:
                    continue
                self.head_weights = {"click": c, "wr": w}
                p = self._val_primary(self.predict(X_val, groups_val))
                if p > best[0] + 1e-5:
                    best = (p, {"click": c, "wr": w})
        self.head_weights = best[1]
        self.info["head_weights"] = best[1]
        self.info["best_val_primary"] = best[0]


class MMoE(_MultiTask):
    def _build(self, meta, X_train):
        trunk = _Trunk(meta, X_train, dim=self.cfg["dim"])
        n_exp, n_task = 4, len(self.tasks)
        self.net = nn.ModuleDict({
            "trunk": trunk,
            "experts": nn.ModuleList(_mlp(trunk.d_out, [64], 64) for _ in range(n_exp)),
            "gates": nn.ModuleList(nn.Linear(trunk.d_out, n_exp) for _ in range(n_task)),
            "towers": nn.ModuleList(_mlp(64, [32], 1) for _ in range(n_task)),
        })

    def _forward(self, X):
        flat = self.net["trunk"].flat(X)
        experts = torch.stack([e(flat) for e in self.net["experts"]], 1)  # (B, E, 64)
        outs = []
        for gate, tower in zip(self.net["gates"], self.net["towers"]):
            w = torch.softmax(gate(flat), dim=1).unsqueeze(-1)
            outs.append(tower((experts * w).sum(1)).squeeze(-1))
        return torch.stack(outs, 1)


class PLE(_MultiTask):
    def _build(self, meta, X_train):
        trunk = _Trunk(meta, X_train, dim=self.cfg["dim"])
        n_task = len(self.tasks)
        self.net = nn.ModuleDict({
            "trunk": trunk,
            "shared": nn.ModuleList(_mlp(trunk.d_out, [64], 64) for _ in range(2)),
            "own": nn.ModuleList(_mlp(trunk.d_out, [64], 64) for _ in range(n_task)),
            "gates": nn.ModuleList(nn.Linear(trunk.d_out, 3) for _ in range(n_task)),
            "towers": nn.ModuleList(_mlp(64, [32], 1) for _ in range(n_task)),
        })

    def _forward(self, X):
        flat = self.net["trunk"].flat(X)
        shared = [e(flat) for e in self.net["shared"]]
        outs = []
        for own, gate, tower in zip(self.net["own"], self.net["gates"], self.net["towers"]):
            experts = torch.stack(shared + [own(flat)], 1)
            w = torch.softmax(gate(flat), dim=1).unsqueeze(-1)
            outs.append(tower((experts * w).sum(1)).squeeze(-1))
        return torch.stack(outs, 1)


class CWM(_TorchRec):
    """Counterfactual watch-time: censored regression of watch ratio, one-sided where
    play_time reaches duration; rank by predicted (duration-normalized) watch time."""

    def _build(self, meta, X_train):
        trunk = _Trunk(meta, X_train, dim=self.cfg["dim"])
        self.net = nn.ModuleDict({"trunk": trunk, "mlp": _mlp(trunk.d_out, [128, 64], 1)})

    def _forward(self, X):
        return self.net["mlp"](self.net["trunk"].flat(X)).squeeze(-1)

    def _loss(self, out, yb, seg, auxb):
        r = (auxb["play_time_ms"] / auxb["duration_ms"].clamp(1)).clamp(0, 1)
        censored = r >= 1.0
        under = torch.relu(r - out)
        return torch.where(censored, under ** 2, (out - r) ** 2).mean()


class DINLite(_TorchRec):
    """Target attention over the seq block (last-20 video/author ids) on top of dcnv2.
    Requires the full_seq spec."""

    def _build(self, meta, X_train):
        trunk = _Trunk(meta, X_train, dim=self.cfg["dim"])
        cols = meta["columns"]
        self.vcols = [cols.index(f"sq_v{k}") for k in range(1, 21)]
        self.acols = [cols.index(f"sq_a{k}") for k in range(1, 21)]
        self.v_emb_i = trunk.cat_idx.index(cols.index("id_video"))
        self.a_emb_i = trunk.cat_idx.index(cols.index("id_author"))
        d = self.cfg["dim"]
        self.net = nn.ModuleDict({
            "trunk": trunk,
            "att": _mlp(4 * d, [32], 1),
            "cross": _CrossNet(trunk.d_out + 2 * d),
            "mlp": _mlp(trunk.d_out + 2 * d, [128], 64),
            "head": nn.Linear(trunk.d_out + 2 * d + 64, 1),
        })
        # seq ids are raw train-vocab codes like id_video/id_author: reuse those tables
        self._vocab_sizes = (trunk.sizes[self.v_emb_i], trunk.sizes[self.a_emb_i])

    def _forward(self, X):
        trunk = self.net["trunk"]
        flat = trunk.flat(X)
        vemb, aemb = trunk.emb[self.v_emb_i], trunk.emb[self.a_emb_i]
        vs, as_ = self._vocab_sizes
        seq_v_raw = X[:, self.vcols].long()
        mask = seq_v_raw < 0
        seq_v = vemb((seq_v_raw + 1).clamp(0, vs - 1))          # (B, 20, d)
        seq_a = aemb((X[:, self.acols].long() + 1).clamp(0, as_ - 1))
        cols = trunk._cols
        tgt_v = vemb((X[:, cols.index("id_video")].long() + 1).clamp(0, vs - 1))
        att_in = torch.cat([seq_v, tgt_v.unsqueeze(1).expand_as(seq_v),
                            seq_v - tgt_v.unsqueeze(1), seq_v * tgt_v.unsqueeze(1)], dim=2)
        w = self.net["att"](att_in).squeeze(-1).masked_fill(mask, -1e9)
        w = torch.softmax(w, dim=1).unsqueeze(-1)
        if bool(mask.all(dim=1).any()):
            w = torch.where(mask.all(dim=1, keepdim=True).unsqueeze(-1), torch.zeros_like(w), w)
        interest = torch.cat([(w * seq_v).sum(1), (w * seq_a).sum(1)], dim=1)
        x = torch.cat([flat, interest], dim=1)
        return self.net["head"](torch.cat([self.net["cross"](x), self.net["mlp"](x)], 1)).squeeze(-1)
