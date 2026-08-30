# scripted: resources

- **iterations_used**: 8
- **scored**: 7
- **kept**: 3
- **reverted**: 4
- **abandoned**: 1
- **wall_clock_hours**: 0.021
- **tokens**: None
- **tokens_note**: not tracked by the harness; agent-side accounting only
- **gpu_hours**: 0.0
- **gpu_note**: nvidia-smi present but not sampled during the run; models ran on CPU
- **interventions**: 0
- **iteration_interpretation**: An iteration is one scored experiment. Failed attempts (max 3) do not consume an iteration number; after 3 the iteration is abandoned. Abandoned iterations count toward the 50 cap but not toward the convergence window, because they produced no score.
