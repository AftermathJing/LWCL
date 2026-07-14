import torch

from lwcl.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = save_checkpoint(
        tmp_path / "last.pt",
        model,
        optimizer,
        scheduler,
        None,
        {"epoch": 1, "global_step": 3},
        {"training": {}},
        trainable_only=False,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)
    payload = load_checkpoint(path, model, optimizer, scheduler, restore_rng=False)
    assert payload["state"]["global_step"] == 3
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
