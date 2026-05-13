import torch

from mymodel.nar.noncausalrefinerwithfuse import NonCausalRefiner


def _run_all_correct_case():
    torch.manual_seed(0)
    model = NonCausalRefiner(
        vocab_size=50,
        input_dim=256,
        hidden_dim=256,
        enable_span_mlm=True,
        span_max_len=3,
        span_max_ratio=0.5,
        span_min_error_tokens=1,
        use_acoustic_preprocess=False,
    )
    model.train()

    hidden = torch.randn(2, 6, 256)
    lengths = torch.tensor([4, 6], dtype=torch.long)
    target = torch.randint(4, 50, (2, 6), dtype=torch.long)
    predict = target.clone()

    target[0, 4:] = -100
    predict[0, 4:] = -100

    res = model(hidden=hidden, target=target, lengths=lengths, predict=predict, compute_mlm_loss=True)
    loss_mlm = res["loss_mlm"]
    assert torch.isfinite(loss_mlm).all(), loss_mlm
    assert float(loss_mlm.detach().cpu()) == 0.0, loss_mlm


def _run_span_padding_safety_case():
    torch.manual_seed(0)
    model = NonCausalRefiner(
        vocab_size=50,
        input_dim=256,
        hidden_dim=256,
        enable_span_mlm=True,
        span_max_len=4,
        span_max_ratio=1.0,
        span_min_error_tokens=1,
        use_acoustic_preprocess=False,
    )
    model.train()

    hidden = torch.randn(2, 6, 256)
    lengths = torch.tensor([4, 4], dtype=torch.long)
    target = torch.randint(4, 50, (2, 6), dtype=torch.long)
    target[:, 4:] = -100

    predict = target.clone()
    predict[0, 1] = (predict[0, 1] + 1) % 50
    predict[0, 2] = (predict[0, 2] + 2) % 50
    predict[1, 3] = (predict[1, 3] + 3) % 50

    res = model(hidden=hidden, target=target, lengths=lengths, predict=predict, compute_mlm_loss=True)
    span_mask = res["span_mask"]
    assert span_mask is not None
    assert span_mask.shape == (2, 6)
    assert span_mask[:, 4:].sum().item() == 0
    assert torch.isfinite(res["loss_mlm"]).all(), res["loss_mlm"]


if __name__ == "__main__":
    _run_all_correct_case()
    _run_span_padding_safety_case()
    print("ok")

