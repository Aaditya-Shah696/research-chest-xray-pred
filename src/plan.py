"""Pure decisions for resume and the time budget. No TensorFlow imports."""

from __future__ import annotations


def should_start_epoch(elapsed_sec: float, last_epoch_sec: float | None, budget_sec: float) -> bool:
    """False when another epoch would run past the session budget.

    The first epoch is allowed even if we do not yet know how long an epoch
    takes. A single epoch is never killed mid-way; the next one is skipped.
    """
    if elapsed_sec >= budget_sec:
        return False
    if last_epoch_sec is None:
        return True
    return elapsed_sec + last_epoch_sec < budget_sec


def infer_phase(
    state: dict,
    stage1_epoch: int,
    stage2_epoch: int,
    stage1_target: int,
    stage2_target: int,
    preds_exist: bool,
) -> str:
    """Return stage1, stage2, predict, or done.

    A time stop does not finish a stage: the next session keeps going.
    Early stopping on stage 2 does finish training.
    """
    early_stage2 = state.get("early_stopped_stage") == 2
    stage2_complete = early_stage2 or (stage2_target > 0 and stage2_epoch >= stage2_target)
    if stage2_complete:
        if preds_exist and state.get("preds_written"):
            return "done"
        return "predict"
    stage1_complete = bool(state.get("stage1_complete")) or (
        stage1_target > 0 and stage1_epoch >= stage1_target
    )
    if stage2_epoch > 0 or stage1_complete:
        return "stage2"
    return "stage1"
