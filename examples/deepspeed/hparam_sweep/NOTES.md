Priority order to run

- E0 Baseline: ds_z3_base_profile.json (measurement only; includes wall_clock_breakdown).
- E1 Mixed precision: ds_z3_e1_bf16.json (force bf16 on H100).
- E2 Overlap + RS: ds_z3_e2_overlap_rs.json (hide gradient/param comm under compute).
- E3 Buckets: ds_z3_e3_buckets_200M.json (200M elements buckets for fewer comm launches).
- E4 Autotune: ds_z3_e4_autotune.json (fast sweep of micro-batch per GPU).
- E5 Persistence threshold: ds_z3_e5_persist_5e5.json (reduce small-message chatter).
- E6 Module granularity: ds_z3_e6_modgran_16.json (lower hook overhead on ZeRO-3).
- E7 Grad accum dtype: ds_z3_e7_gradaccum_bf16.json (keep accumulation in bf16).
- E9 Quantized grads: ds_z3_e9_quant_grads.json (ZeRO++ comm reduction; keep if stable).
- E10 Allgather A/B: ds_z3_e10_allgather_partitions_false.json (compare vs true).
- E11 Activation ckpt: ds_z3_e11_act_ckpt.json (only if memory-bound at 32k; may enable larger micro-batch).

H100-specific notes

- Prefer bf16 (faster and more stable than fp16 on Hopper).
- NVLink on single node supports larger buckets (200M is a good starting point). If step time is still comm-limited, try 500M; if latency-limited,
try 100M.
- At seq len 32k, activation memory is huge. If you can’t fit a reasonable micro-batch, enable ds_z3_e11_act_ckpt.json or enable gradient checkpointing
in your training script to trade compute for memory.
- If available in your stack, enable FlashAttention-2 in the model config to cut attention memory/time substantially on long contexts.
