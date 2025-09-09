import math
import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("TRANSFORMERS_AVAILABLE", "1") == "0",
    reason="Transformers not available in test environment",
)


def _hf_available():
    try:
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _hf_available(), reason="requires transformers")
def test_dynamic_yarn_matches_static_endpoints():
    import torch
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    from llamafactory.model.model_utils.dynamic_rope import DynamicYarnRotaryEmbedding

    torch.manual_seed(0)

    N0 = 512
    s_max = 4.0
    head_dim = 64
    cfg = LlamaConfig(
        hidden_size=head_dim,
        intermediate_size=4 * head_dim,
        num_attention_heads=1,
        num_hidden_layers=1,
        max_position_embeddings=int(N0 * s_max),
        rope_scaling={
            "rope_type": "yarn",
            "factor": s_max,
            "original_max_position_embeddings": N0,
            "dynamic": True,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    )

    # Base/static modules for comparison
    base_cfg_1 = cfg.__class__(
        **{
            **cfg.to_dict(),
            "rope_scaling": {**cfg.rope_scaling, "factor": 1.0, "dynamic": False},
            "max_position_embeddings": N0,
        }
    )
    base_cfg_smax = cfg.__class__(
        **{
            **cfg.to_dict(),
            "rope_scaling": {**cfg.rope_scaling, "factor": s_max, "dynamic": False},
            "max_position_embeddings": int(N0 * s_max),
        }
    )
    static_1 = LlamaRotaryEmbedding(base_cfg_1)
    static_smax = LlamaRotaryEmbedding(base_cfg_smax)

    # Dynamic wrapper (wrap a base module instance for class reference)
    base_module = LlamaRotaryEmbedding(base_cfg_1)
    dyn = DynamicYarnRotaryEmbedding(base_module, cfg, quant_step=0.05)

    # helpers
    def run(m, L, dtype=torch.float32):
        x = torch.zeros((1, L, 1, head_dim), dtype=dtype)
        pos = torch.arange(L, dtype=torch.long).unsqueeze(0)
        cos, sin = m(x, pos)
        return cos.detach(), sin.detach()

    # Short context: behaves like static factor=1
    cos_d, sin_d = run(dyn, int(N0 // 2))
    cos_1, sin_1 = run(static_1, int(N0 // 2))
    assert torch.allclose(cos_d, cos_1, atol=1e-5, rtol=1e-5)
    assert torch.allclose(sin_d, sin_1, atol=1e-5, rtol=1e-5)

    # Long context near max: behaves like static factor=s_max
    cos_d2, sin_d2 = run(dyn, int(N0 * s_max) - 8)
    cos_s, sin_s = run(static_smax, int(N0 * s_max) - 8)
    assert torch.allclose(cos_d2, cos_s, atol=1e-5, rtol=1e-5)
    assert torch.allclose(sin_d2, sin_s, atol=1e-5, rtol=1e-5)

    # Mid context: quantized to 0.05 step
    mid_L = int(N0 * 1.52)
    # Expected effective scale rounded to nearest 0.05
    expected_scale = round((mid_L / N0) / 0.05) * 0.05
    base_cfg_mid = cfg.__class__(
        **{
            **cfg.to_dict(),
            "rope_scaling": {**cfg.rope_scaling, "factor": float(expected_scale), "dynamic": False},
            "max_position_embeddings": int(math.ceil(N0 * float(expected_scale))),
        }
    )
    static_mid = LlamaRotaryEmbedding(base_cfg_mid)
    cos_dm, sin_dm = run(dyn, mid_L)
    cos_m, sin_m = run(static_mid, mid_L)
    assert torch.allclose(cos_dm, cos_m, atol=1e-5, rtol=1e-5)
    assert torch.allclose(sin_dm, sin_m, atol=1e-5, rtol=1e-5)
