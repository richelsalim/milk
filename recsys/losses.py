"""Ranking losses for the torch models. All within-user losses take `seg`: an int64
tensor of consecutive segment ids (0..S-1), one per row, marking which user a row
belongs to inside the batch."""

import torch
import torch.nn.functional as F


def bce(logits, y, seg=None, w=None):
    return F.binary_cross_entropy_with_logits(logits, y, weight=w)


def _segment_max(values, seg, n_seg):
    out = torch.full((n_seg,), -1e30, device=values.device)
    return out.scatter_reduce(0, seg, values, reduce="amax", include_self=True)


def _segment_sum(values, seg, n_seg):
    return torch.zeros(n_seg, device=values.device).scatter_add(0, seg, values)


def listwise_softmax_within_user(logits, y, seg):
    """-sum_u sum_i (y_i / sum_u y) * log softmax_u(logits)_i, over users with >=1 positive."""
    n_seg = int(seg.max()) + 1
    m = _segment_max(logits, seg, n_seg)
    z = logits - m[seg]
    logsumexp = torch.log(_segment_sum(z.exp(), seg, n_seg) + 1e-12)
    logp = z - logsumexp[seg]
    pos_per_seg = _segment_sum(y, seg, n_seg)
    weight = y / (pos_per_seg[seg] + 1e-12)
    per_seg = _segment_sum(-weight * logp * (pos_per_seg[seg] > 0), seg, n_seg)
    active = (pos_per_seg > 0).sum()
    return per_seg.sum() / (active + 1e-12)


def bpr_pairwise_within_user(logits, y, seg):
    """-log sigmoid(pos - neg) over one random pos/neg pair per eligible user."""
    n = len(logits)
    n_seg = int(seg.max()) + 1
    # exact per-segment random argmax: int64 composite (random << 21) | row_index
    idx = torch.arange(n, device=logits.device)
    val = (torch.randint(0, 1 << 20, (n,), device=logits.device) << 21) | idx
    neg_one = torch.full((n,), -1, dtype=torch.int64, device=logits.device)

    def pick(eligible):
        cand = torch.where(eligible, val, neg_one)
        best = torch.full((n_seg,), -1, dtype=torch.int64, device=logits.device)
        best = best.scatter_reduce(0, seg, cand, reduce="amax", include_self=True)
        return best

    pos_best, neg_best = pick(y > 0.5), pick(y < 0.5)
    ok = (pos_best >= 0) & (neg_best >= 0)
    if not ok.any():
        return logits.sum() * 0.0
    mask = (1 << 21) - 1
    return -F.logsigmoid(logits[pos_best[ok] & mask] - logits[neg_best[ok] & mask]).mean()


def mixed(weight=0.5):
    """weight * bce + (1 - weight) * listwise."""
    def loss(logits, y, seg):
        return weight * bce(logits, y) + (1 - weight) * listwise_softmax_within_user(logits, y, seg)
    return loss


LOSSES = {"bce": bce, "bpr": bpr_pairwise_within_user,
          "listwise": listwise_softmax_within_user, "mixed": mixed(0.5)}
