Distributed Checkpointing (DCP) Notes

- Process group: Uses the data-parallel `Accelerate` process group when available to ensure consistent sharding and coordination across ranks.
- Barriers: Synchronizes before and after DCP save/load to line up ranks cleanly.
- NCCL robustness: Consider exporting `NCCL_ASYNC_ERROR_HANDLING=1` for more robust collective error detection and recovery. Ensure communicator consistency across save/load (same world size and rank ordering).
- DeepSpeed: DeepSpeed save/load remains separate via `engine.save_checkpoint` and is guarded to avoid stale torch.compile captures.
- Model-only saves: When `save_only_model` is true or a non-standard optimizer (paged/8-bit) is detected, optimizer/scheduler states are skipped. Warnings clarify when optimizer/scheduler restore is unavailable.
