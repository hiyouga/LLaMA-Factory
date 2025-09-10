DeepSpeed Example Configs

This folder contains minimal DeepSpeed configuration examples for common setups (Zeo-0/2/3, with or without CPU offload).

Logging options
- DeepSpeed provides several logging/timing features that can add overhead or require CUDA event timing:
  - `steps_per_print`, `wall_clock_breakdown`
  - `comms_logger` and `comms_config`
  - `monitor_config` (TensorBoard/Comet/W&B/CSV)
  - `flops_profiler`

Important: Avoid these with `torch.compile`
- When using `torch.compile`, we recommend avoiding DeepSpeed’s timing and logging features above. CUDA event timing used by these features can be brittle under compiled graphs and may cause runtime errors in some environments.

Recommendations
- Prefer Hugging Face Trainer logging on Rank 0 for general metrics.
- If you need DeepSpeed-specific statistics, enable them only when not using `torch.compile`.
- For performance insights, consider lightweight, external profiling tools rather than enabling FLOPs or comm profilers in production runs.

Notes
- These example JSON files intentionally omit DeepSpeed logging/timing fields to provide stable defaults.
