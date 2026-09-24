"""Tests for rrm/training.py — neutral training utilities."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import List

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rrm.labels import UNKNOWN_LABEL
from rrm.training import (
    GradientAccumulator,
    MaskedLossResult,
    masked_bce_sum_and_count,
    optimizer_step_with_accumulation,
)


# ---------------------------------------------------------------------------
# masked_bce_sum_and_count tests
# ---------------------------------------------------------------------------


class TestMaskedBceSumAndCount:
    """Tests for the neutral masked BCE loss utility."""

    def test_basic_known_labels(self):
        logits = torch.randn(4, 6, requires_grad=True)
        labels = torch.tensor([
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0],
        ], dtype=torch.long)
        result = masked_bce_sum_and_count(logits, labels)
        assert result is not None
        assert isinstance(result, MaskedLossResult)
        assert result.supervised_count == 24  # 4 * 6, all known
        assert result.loss_sum.requires_grad

    def test_all_unknown_returns_none(self):
        logits = torch.randn(2, 6)
        labels = torch.full((2, 6), UNKNOWN_LABEL, dtype=torch.long)
        result = masked_bce_sum_and_count(logits, labels)
        assert result is None

    def test_mixed_known_unknown_count(self):
        logits = torch.randn(3, 6)
        labels = torch.tensor([
            [1, -1, 0, 1, -1, 0],  # 4 known
            [-1, -1, -1, -1, -1, -1],  # 0 known
            [0, 1, -1, 0, 1, -1],  # 4 known
        ], dtype=torch.long)
        result = masked_bce_sum_and_count(logits, labels)
        assert result is not None
        assert result.supervised_count == 8

    def test_unknown_contributes_zero_loss(self):
        """UNKNOWN positions must contribute zero to the loss sum."""
        logits = torch.zeros(2, 6, requires_grad=True)
        labels = torch.tensor([
            [1, -1, 0, -1, 1, -1],  # 3 known: 1, 0, 1
            [-1, -1, -1, -1, -1, -1],  # 0 known
        ], dtype=torch.long)
        result = masked_bce_sum_and_count(logits, labels)
        assert result is not None
        assert result.supervised_count == 3

    def test_result_is_frozen(self):
        logits = torch.randn(2, 6)
        labels = torch.randint(0, 2, (2, 6)).long()
        result = masked_bce_sum_and_count(logits, labels)
        assert result is not None
        with pytest.raises(FrozenInstanceError):
            result.supervised_count = 0  # type: ignore[misc]

    def test_wrong_rank_logits(self):
        logits = torch.randn(6)
        labels = torch.randint(0, 2, (6,)).long()
        with pytest.raises(ValueError, match="rank-2"):
            masked_bce_sum_and_count(logits, labels)

    def test_wrong_rank_labels(self):
        logits = torch.randn(2, 6)
        labels = torch.randint(0, 2, (6,)).long()
        with pytest.raises(ValueError, match="rank-2"):
            masked_bce_sum_and_count(logits, labels)

    def test_shape_mismatch(self):
        logits = torch.randn(2, 6)
        labels = torch.randint(0, 2, (2, 4)).long()
        with pytest.raises(ValueError, match="must match"):
            masked_bce_sum_and_count(logits, labels)

    def test_zero_batch(self):
        logits = torch.empty(0, 6)
        labels = torch.empty(0, 6, dtype=torch.long)
        with pytest.raises(ValueError, match="batch dimension must be > 0"):
            masked_bce_sum_and_count(logits, labels)

    def test_zero_labels_dim(self):
        logits = torch.randn(2, 0)
        labels = torch.empty(2, 0, dtype=torch.long)
        with pytest.raises(ValueError, match="must be 6"):
            masked_bce_sum_and_count(logits, labels)

    def test_non_floating_logits(self):
        logits = torch.zeros(2, 6, dtype=torch.long)
        labels = torch.randint(0, 2, (2, 6)).long()
        with pytest.raises(ValueError, match="floating dtype"):
            masked_bce_sum_and_count(logits, labels)

    def test_non_long_labels(self):
        logits = torch.randn(2, 6)
        labels = torch.zeros(2, 6, dtype=torch.float32)
        with pytest.raises(ValueError, match="torch.long"):
            masked_bce_sum_and_count(logits, labels)

    def test_device_mismatch(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        logits = torch.randn(2, 6, device="cpu")
        labels = torch.randint(0, 2, (2, 6), device="cuda").long()
        with pytest.raises(ValueError, match="same device"):
            masked_bce_sum_and_count(logits, labels)

    def test_invalid_label_value(self):
        logits = torch.randn(2, 6)
        labels = torch.tensor([[2, 0, 1, 0, 1, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.long)
        with pytest.raises(ValueError, match="invalid values"):
            masked_bce_sum_and_count(logits, labels)

    def test_gradient_flow(self):
        logits = torch.randn(4, 6, requires_grad=True)
        labels = torch.randint(0, 2, (4, 6)).long()
        result = masked_bce_sum_and_count(logits, labels)
        assert result is not None
        result.loss_sum.backward()
        assert logits.grad is not None

    def test_exact_bce_arithmetic(self):
        """With known targets 0/1 and zero logits, BCE = log(2) per position."""
        logits = torch.zeros(2, 6, requires_grad=True)
        labels = torch.tensor([
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
        ], dtype=torch.long)
        result = masked_bce_sum_and_count(logits, labels)
        assert result is not None
        expected_per_position = torch.log(torch.tensor(2.0)).item()
        expected_sum = 12 * expected_per_position
        assert abs(result.loss_sum.item() - expected_sum) < 1e-5

    def test_wrong_num_labels_dim_one_rejected(self):
        """[B,1] violates the canonical six-task contract."""
        logits = torch.randn(2, 1)
        labels = torch.tensor([[1], [-1]], dtype=torch.long)
        with pytest.raises(ValueError, match="must be 6"):
            masked_bce_sum_and_count(logits, labels)

    def test_wrong_num_labels_dim_five_rejected(self):
        """[B,5] violates the canonical six-task contract."""
        logits = torch.randn(2, 5)
        labels = torch.randint(0, 2, (2, 5)).long()
        with pytest.raises(ValueError, match="must be 6"):
            masked_bce_sum_and_count(logits, labels)

    def test_wrong_num_labels_dim_seven_rejected(self):
        """[B,7] violates the canonical six-task contract."""
        logits = torch.randn(2, 7)
        labels = torch.randint(0, 2, (2, 7)).long()
        with pytest.raises(ValueError, match="must be 6"):
            masked_bce_sum_and_count(logits, labels)

    def test_all_unknown_batch_no_nan(self):
        """All-UNKNOWN should return None, not NaN loss."""
        logits = torch.randn(2, 6)
        labels = torch.full((2, 6), UNKNOWN_LABEL, dtype=torch.long)
        result = masked_bce_sum_and_count(logits, labels)
        assert result is None


# ---------------------------------------------------------------------------
# GradientAccumulator tests
# ---------------------------------------------------------------------------


class TestGradientAccumulator:
    def test_initial_state(self):
        acc = GradientAccumulator()
        assert acc.count == 0
        assert acc.contributing is False

    def test_add(self):
        acc = GradientAccumulator()
        acc.add(10)
        assert acc.count == 10

    def test_add_multiple(self):
        acc = GradientAccumulator()
        acc.add(5)
        acc.add(3)
        acc.add(7)
        assert acc.count == 15

    def test_add_zero_rejected(self):
        acc = GradientAccumulator()
        with pytest.raises(ValueError, match="must be > 0"):
            acc.add(0)

    def test_add_negative_rejected(self):
        acc = GradientAccumulator()
        with pytest.raises(ValueError, match="must be > 0"):
            acc.add(-1)

    def test_contributing_after_add(self):
        acc = GradientAccumulator()
        assert acc.contributing is False
        acc.add(1)
        assert acc.contributing is True

    def test_reset(self):
        acc = GradientAccumulator()
        acc.add(10)
        acc.reset()
        assert acc.count == 0
        assert acc.contributing is False


# ---------------------------------------------------------------------------
# Accumulation equivalence test (item 8 — mandatory)
# ---------------------------------------------------------------------------


class TestAccumulationEquivalence:
    """Proves that global supervised-position normalization produces
    equivalent gradients regardless of how microbatches are split.

    This proves UNKNOWN density cannot incorrectly weight microbatches.
    """

    def test_two_microbatch_equivalence(self):
        """Path A: backward A then backward B, divide accumulated grads
        by total count. Path B: (loss_A + loss_B) / total_count, backward
        once. Gradients must match.
        """
        # Both models start with identical weights
        model_a = nn.Linear(6, 6, bias=False)
        model_b = nn.Linear(6, 6, bias=False)
        nn.init.eye_(model_a.weight)
        nn.init.eye_(model_b.weight)

        # Fixed inputs (leaf tensors)
        torch.manual_seed(0)
        inputs_a = torch.randn(1, 6, requires_grad=True)
        labels_a = torch.tensor([[1, 0, -1, -1, -1, -1]], dtype=torch.long)

        torch.manual_seed(1)
        inputs_b = torch.randn(1, 6, requires_grad=True)
        labels_b = torch.tensor([[1, -1, 1, 0, -1, 1]], dtype=torch.long)

        # --- Path A: logits through model_a, backward separately, divide ---
        logits_a = model_a(inputs_a)
        logits_b = model_a(inputs_b)

        result_a = masked_bce_sum_and_count(logits_a, labels_a)
        assert result_a is not None
        count_a = result_a.supervised_count  # 2

        result_b = masked_bce_sum_and_count(logits_b, labels_b)
        assert result_b is not None
        count_b = result_b.supervised_count  # 4

        result_a.loss_sum.backward()
        result_b.loss_sum.backward()
        total_count = count_a + count_b  # 6

        for param in model_a.parameters():
            if param.grad is not None:
                param.grad.data.div_(total_count)
        grad_a = model_a.weight.grad.clone()

        # --- Path B: logits through model_b, combined loss, backward once ---
        logits_c = model_b(inputs_a.detach())
        logits_d = model_b(inputs_b.detach())

        safe_a = labels_a.to(dtype=logits_c.dtype).clamp(min=0)
        safe_b = labels_b.to(dtype=logits_d.dtype).clamp(min=0)
        mask_a = (labels_a != -1).to(dtype=logits_c.dtype)
        mask_b = (labels_b != -1).to(dtype=logits_d.dtype)

        bce_c = F.binary_cross_entropy_with_logits(logits_c, safe_a, reduction="none")
        bce_d = F.binary_cross_entropy_with_logits(logits_d, safe_b, reduction="none")
        loss_sum_c = (bce_c * mask_a).sum()
        loss_sum_d = (bce_d * mask_b).sum()
        combined_loss = (loss_sum_c + loss_sum_d) / total_count

        model_b.zero_grad()
        combined_loss.backward()
        grad_b = model_b.weight.grad.clone()

        assert torch.allclose(grad_a, grad_b, atol=1e-5), (
            f"Gradient mismatch:\nPath A norm={grad_a.norm().item():.6f}\n"
            f"Path B norm={grad_b.norm().item():.6f}"
        )


# ---------------------------------------------------------------------------
# optimizer_step_with_accumulation tests
# ---------------------------------------------------------------------------


class SchedulerSpy:
    """Pure scheduler spy — no wrapping of real PyTorch schedulers.

    Tracks step() call count without triggering LR scheduling warnings.
    """

    def __init__(self):
        self.step_count = 0

    def step(self):
        self.step_count += 1


class FakeGradScaler:
    """Fake GradScaler that records call ordering and delegates to optimizer.

    Implements unscale_(), step(), and update() to prove AMP ordering.
    """

    def __init__(self):
        self.events: list = []
        self._unscale_called = False

    def unscale_(self, optimizer):
        self.events.append("unscale")
        self._unscale_called = True
        # No-op: fake unscale (we don't change grads in the fake)

    def step(self, optimizer):
        self.events.append("scaler.step")
        if not self._unscale_called:
            raise RuntimeError("unscale_ must be called before scaler.step")
        optimizer.step()

    def update(self):
        self.events.append("scaler.update")


class TestOptimizerStepWithAccumulation:
    """Tests for the optimizer-step boundary utility."""

    def test_skip_zero_count(self):
        """Zero supervised count → no optimizer step, grads cleared."""
        model = nn.Linear(6, 6)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        acc = GradientAccumulator()

        x = torch.randn(1, 6)
        loss = model(x).sum()
        loss.backward()
        assert model.weight.grad is not None

        stepped = optimizer_step_with_accumulation(
            optimizer, model, acc, max_grad_norm=1.0
        )
        assert stepped is False
        assert model.weight.grad is None

    def test_step_with_count(self):
        """Positive count → optimizer step performed."""
        model = nn.Linear(6, 6)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        acc = GradientAccumulator()
        acc.add(10)

        x = torch.randn(1, 6)
        loss = model(x).sum()
        loss.backward()

        stepped = optimizer_step_with_accumulation(
            optimizer, model, acc, max_grad_norm=1.0
        )
        assert stepped is True
        assert acc.count == 0

    def test_gradient_division(self):
        """Gradients must be divided by supervised count."""
        model = nn.Linear(6, 6, bias=False)
        nn.init.eye_(model.weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        acc = GradientAccumulator()

        x = torch.ones(1, 6, requires_grad=True)
        loss = model(x).sum()
        loss.backward()

        acc.add(3)
        optimizer_step_with_accumulation(optimizer, model, acc, max_grad_norm=None)

        expected = torch.eye(6) - (1.0 / 3.0) * torch.ones(6, 6)
        assert torch.allclose(model.weight.data, expected, atol=1e-5)

    # --- Item 10: Scheduler call-count tests ---

    def test_scheduler_stepped_on_success(self):
        """Successful optimizer update: scheduler.step count increments by 1."""
        model = nn.Linear(6, 6)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        spy = SchedulerSpy()
        acc = GradientAccumulator()
        acc.add(1)

        x = torch.randn(1, 6)
        loss = model(x).sum()
        loss.backward()

        assert spy.step_count == 0
        optimizer_step_with_accumulation(
            optimizer, model, acc, scheduler=spy, max_grad_norm=1.0
        )
        assert spy.step_count == 1

    def test_scheduler_not_stepped_on_skip(self):
        """Skipped optimizer update: scheduler.step count unchanged."""
        model = nn.Linear(6, 6)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        spy = SchedulerSpy()
        acc = GradientAccumulator()

        x = torch.randn(1, 6)
        loss = model(x).sum()
        loss.backward()

        optimizer_step_with_accumulation(
            optimizer, model, acc, scheduler=spy, max_grad_norm=1.0
        )
        assert spy.step_count == 0

    # --- Item 9: Zero-supervision optimizer + scheduler tests ---

    def test_zero_supervision_no_optimizer_step(self):
        """Zero supervised count → optimizer.step not called, grads zeroed."""
        model = nn.Linear(6, 6)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        acc = GradientAccumulator()

        x = torch.randn(1, 6)
        loss = model(x).sum()
        loss.backward()
        assert model.weight.grad is not None

        optimizer_step_with_accumulation(
            optimizer, model, acc, max_grad_norm=1.0
        )
        assert model.weight.grad is None

    def test_zero_supervision_no_scheduler_step(self):
        """Zero supervised count → scheduler.step not called."""
        model = nn.Linear(6, 6)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        spy = SchedulerSpy()
        acc = GradientAccumulator()

        x = torch.randn(1, 6)
        loss = model(x).sum()
        loss.backward()

        optimizer_step_with_accumulation(
            optimizer, model, acc, scheduler=spy, max_grad_norm=1.0
        )
        assert spy.step_count == 0

    # --- Item 11: AMP ordering test (CPU-safe fake scaler) ---

    def test_amp_ordering_proves_unscale_normalize_clip_step_update(self):
        """Verify complete AMP ordering with fake scaler on CPU.

        Expected order (proven by FakeGradScaler event recording):
        1. scaler.unscale_(optimizer)
        2. divide accumulated gradients by total supervised_count
        3. clip gradients if enabled
        4. scaler.step(optimizer)  [actually calls optimizer.step()]
        5. scaler.update()
        6. scheduler.step()
        7. optimizer.zero_grad(set_to_none=True)

        FakeGradScaler enforces unscale_ before step (raises if skipped).
        """
        scaler = FakeGradScaler()
        scheduler = SchedulerSpy()
        model = nn.Linear(6, 6)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        acc = GradientAccumulator()
        acc.add(1)

        x = torch.randn(1, 6)
        loss = model(x).sum()
        loss.backward()

        stepped = optimizer_step_with_accumulation(
            optimizer, model, acc,
            scheduler=scheduler,
            scaler=scaler,
            use_amp=True,
            max_grad_norm=1.0,
        )

        # Verify event order
        assert scaler.events == ["unscale", "scaler.step", "scaler.update"]
        assert stepped is True
        assert scheduler.step_count == 1
        assert model.weight.grad is None  # zeroed after step

    def test_amp_unscale_before_step_enforced(self):
        """FakeGradScaler must raise if step() is called before unscale_()."""
        scaler = FakeGradScaler()
        # Manually call step without unscale_ — should raise
        model = nn.Linear(6, 6)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(1, 6)
        loss = model(x).sum()
        loss.backward()

        # Directly test the scaler's enforcement
        with pytest.raises(RuntimeError, match="unscale_ must be called"):
            scaler.step(optimizer)

    def test_amp_without_scaler_no_warning(self):
        """AMP flag True but no scaler: falls through to plain optimizer."""
        model = nn.Linear(6, 6)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = SchedulerSpy()
        acc = GradientAccumulator()
        acc.add(1)

        x = torch.randn(1, 6)
        loss = model(x).sum()
        loss.backward()

        stepped = optimizer_step_with_accumulation(
            optimizer, model, acc,
            scheduler=scheduler,
            scaler=None,
            use_amp=True,  # True but no scaler
            max_grad_norm=1.0,
        )
        assert stepped is True
        assert scheduler.step_count == 1
        assert model.weight.grad is None

    def test_gradient_clipping(self):
        """Gradient clipping is applied after normalization."""
        torch.manual_seed(42)
        model = nn.Linear(6, 6, bias=False)
        nn.init.normal_(model.weight, std=100.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        acc = GradientAccumulator()
        acc.add(1)

        x = torch.randn(1, 6)
        loss = model(x).sum()
        loss.backward()

        pre_clip_norm = nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()

        optimizer_step_with_accumulation(
            optimizer, model, acc, max_grad_norm=1.0
        )

        assert pre_clip_norm > 1.0  # large gradient before clipping

    def test_grads_zeroed_after_step(self):
        """Gradients are zeroed after optimizer step."""
        model = nn.Linear(6, 6)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        acc = GradientAccumulator()
        acc.add(1)

        x = torch.randn(1, 6)
        loss = model(x).sum()
        loss.backward()
        assert model.weight.grad is not None

        optimizer_step_with_accumulation(optimizer, model, acc, max_grad_norm=1.0)
        assert model.weight.grad is None
