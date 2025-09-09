import sys

import torch

from llamafactory.train.trainer_utils import cce_loss_func


def _standard_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    vocab = logits.size(-1)
    labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:].contiguous().view(-1).to(logits.device)
    logits = logits.view(-1, vocab).float()
    return torch.nn.functional.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="mean")


def test_cce_fallback_matches_ce():
    torch.manual_seed(0)
    bsz, seqlen, vocab = 2, 5, 7
    logits = torch.randn(bsz, seqlen, vocab)
    labels = torch.randint(low=0, high=vocab, size=(bsz, seqlen))
    # mask some tokens
    labels[0, 0] = -100
    outputs = {"logits": logits}

    loss_cce = cce_loss_func(outputs, labels)
    loss_std = _standard_ce(logits, labels)

    assert torch.allclose(loss_cce, loss_std, atol=1e-6), (loss_cce.item(), loss_std.item())


def test_cce_optional_normalization():
    torch.manual_seed(0)
    bsz, seqlen, vocab = 2, 4, 6
    logits = torch.randn(bsz, seqlen, vocab)
    labels = torch.randint(low=0, high=vocab, size=(bsz, seqlen))
    outputs = {"logits": logits}

    base = cce_loss_func(outputs, labels)
    scaled = cce_loss_func(outputs, labels, num_items_in_batch=torch.tensor(bsz * seqlen))

    # scaled should be close to base when normalization matches number of elements
    assert torch.allclose(base, scaled, atol=1e-6)


def test_cce_library_path_is_graceful(monkeypatch):
    """Simulate presence of the library to ensure function still returns a tensor.

    We don't validate equivalence here; just ensure the import path doesn't raise.
    """

    class _DummyCCE:
        @staticmethod
        def linear_cross_entropy(hidden, classifier, labels, ignore_index=-100, reduction="mean"):
            # Emulate library path by computing CE via matmul(head) -> logits
            logits = hidden @ classifier.t()
            return torch.nn.functional.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    monkeypatch.setitem(sys.modules, "cut_cross_entropy", _DummyCCE)

    torch.manual_seed(0)
    bsz, seqlen, vocab, dim = 2, 5, 8, 3
    # fabricate embeddings and classifier and attach to outputs to trigger library path
    embeddings = torch.randn(bsz, seqlen, dim)
    classifier = torch.randn(vocab, dim)

    class LMHead:
        def __init__(self, w):
            self.weight = w

    class DummyModel:
        def named_modules(self):
            return [("lm_head", LMHead(classifier))]

    labels = torch.randint(low=0, high=vocab, size=(bsz, seqlen))
    outputs = {"logits": torch.randn(bsz, seqlen, vocab), "hidden_states": embeddings, "model": DummyModel()}

    loss = cce_loss_func(outputs, labels)
    assert torch.is_tensor(loss) and loss.ndim == 0
