Usage:

- `pretrain.sh`: do pre-train (optional)
- `sft.sh`: do supervised fine-tune
- `reward.sh`: do reward modeling (must after sft.sh)
- `ppo.sh`: do PPO training (must after sft.sh and reward.sh)
- `dpo.sh`: do DPO training (must after sft.sh)
- `predict.sh`: do predict (must after sft.sh and dpo.sh)
