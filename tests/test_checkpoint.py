from __future__ import annotations

from pathlib import Path

import torch

from xtbflow.training.checkpoint import load_joint_checkpoint, save_joint_checkpoint


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 1, bias=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


def _train_step(model: TinyModel, optimizer: torch.optim.Optimizer, value: torch.Tensor, target: torch.Tensor) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = (model(value) - target).square().mean()
    loss.backward()
    optimizer.step()


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_nested_equal(first, second)
    else:
        assert left == right


def test_training_checkpoint_restores_optimizer_loop_and_rng_for_continuation(tmp_path: Path):
    torch.manual_seed(17)
    value = torch.tensor([[0.4, -0.2]], dtype=torch.float64)
    target = torch.tensor([[0.7]], dtype=torch.float64)
    continuous = TinyModel().double()
    continuous_optimizer = torch.optim.Adam(continuous.parameters(), lr=0.03)
    for _ in range(2):
        _train_step(continuous, continuous_optimizer, value, target)
    checkpoint = tmp_path / "training.pt"
    save_joint_checkpoint(
        checkpoint,
        continuous,
        continuous_optimizer,
        loop={"global_step": 2, "data_order": [0, 1]},
        sampler_state={"next_batch": 2},
    )
    _train_step(continuous, continuous_optimizer, value, target)

    restored = TinyModel().double()
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.03)
    info = load_joint_checkpoint(checkpoint, restored, restored_optimizer)
    assert info["resume_kind"] == "training"
    assert info["complete_resume"] is True
    _train_step(restored, restored_optimizer, value, target)
    for name, parameter in continuous.state_dict().items():
        torch.testing.assert_close(parameter, restored.state_dict()[name])
    _assert_nested_equal(continuous_optimizer.state_dict(), restored_optimizer.state_dict())
    no_rng_model = TinyModel().double()
    no_rng_optimizer = torch.optim.Adam(no_rng_model.parameters(), lr=0.03)
    no_rng_info = load_joint_checkpoint(checkpoint, no_rng_model, no_rng_optimizer, restore_rng=False)
    assert no_rng_info["resume_kind"] == "incomplete_training"
    assert no_rng_info["complete_resume"] is False


def test_checkpoint_kinds_do_not_promote_inference_or_incomplete_training(tmp_path: Path):
    model = TinyModel()
    inference_path = tmp_path / "inference.pt"
    save_joint_checkpoint(inference_path, model)
    inference_info = load_joint_checkpoint(inference_path, TinyModel())
    assert inference_info["resume_kind"] == "inference"
    assert inference_info["complete_resume"] is False

    incomplete_path = tmp_path / "incomplete.pt"
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    save_joint_checkpoint(incomplete_path, model, optimizer)
    incomplete_info = load_joint_checkpoint(incomplete_path, TinyModel(), torch.optim.SGD(TinyModel().parameters(), lr=0.1))
    assert incomplete_info["resume_kind"] == "incomplete_training"
    assert incomplete_info["complete_resume"] is False


def test_legacy_checkpoint_is_explicitly_incomplete(tmp_path: Path):
    model = TinyModel()
    path = tmp_path / "legacy.pt"
    torch.save({"model": model.state_dict(), "optimizer": None, "seed": 3}, path)
    info = load_joint_checkpoint(path, TinyModel())
    assert info["resume_kind"] == "legacy_incomplete"
    assert info["complete_resume"] is False
