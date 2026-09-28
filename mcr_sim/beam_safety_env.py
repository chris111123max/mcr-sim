"""MCREnv integration for production Beam/SDF unilateral safety.

The physical constraint is always executed on every physics substep. The
validation mode controls only the independent post-commit 10 um verifier:

- debug: verify every physics substep;
- audit: verify every active or near-wall substep, plus periodic samples,
  reset, and terminal states;
- off: disable only the independent verifier. The physical unilateral
  constraint remains active.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .beam_sdf_unilateral_fast import (
    ACTIVE_BAND_M,
    DENSE_SPACING_M,
    FastBeamUnilateralPlanner,
    LiveBeamSDFAdapter,
    MARGIN_M,
    _as_array,
    load_plugin,
    resolve_plugin_path,
)
from .mcr_rl_env import MCREnv


DEFAULT_COMMITTED_PENETRATION_LIMIT_M = 1.0e-6


@dataclass(frozen=True)
class BeamSafetyValidationConfig:
    mode: str = "audit"
    audit_interval_substeps: int = 16
    audit_near_clearance_m: float = MARGIN_M + ACTIVE_BAND_M
    committed_penetration_limit_m: float = DEFAULT_COMMITTED_PENETRATION_LIMIT_M

    def normalized(self) -> "BeamSafetyValidationConfig":
        mode = str(self.mode).strip().lower()
        if mode not in ("debug", "audit", "off"):
            raise ValueError("beam safety validation mode must be debug/audit/off")
        interval = int(self.audit_interval_substeps)
        if interval < 1:
            raise ValueError("audit_interval_substeps must be >= 1")
        near = float(self.audit_near_clearance_m)
        if not np.isfinite(near) or near < MARGIN_M:
            raise ValueError("audit_near_clearance_m must be >= requested margin")
        limit = float(self.committed_penetration_limit_m)
        if not np.isfinite(limit) or limit < 0.0:
            raise ValueError("committed_penetration_limit_m must be non-negative")
        return BeamSafetyValidationConfig(
            mode=mode,
            audit_interval_substeps=interval,
            audit_near_clearance_m=near,
            committed_penetration_limit_m=limit,
        )


class BeamSafetyMCREnv(MCREnv):
    """MCREnv with Fast-FD Beam/SDF unilateral safety attached after reset."""

    def __init__(
        self,
        *args,
        beam_safety_plugin: str | Path | None = None,
        beam_safety_validation_mode: str = "audit",
        beam_safety_audit_interval_substeps: int = 16,
        beam_safety_audit_near_clearance_m: float = MARGIN_M + ACTIVE_BAND_M,
        beam_safety_committed_penetration_limit_m: float = (
            DEFAULT_COMMITTED_PENETRATION_LIMIT_M
        ),
        **kwargs,
    ):
        self._beam_validation = BeamSafetyValidationConfig(
            mode=beam_safety_validation_mode,
            audit_interval_substeps=beam_safety_audit_interval_substeps,
            audit_near_clearance_m=beam_safety_audit_near_clearance_m,
            committed_penetration_limit_m=(
                beam_safety_committed_penetration_limit_m
            ),
        ).normalized()
        self._beam_plugin_path = resolve_plugin_path(beam_safety_plugin)
        self._beam_plugin_handle = load_plugin(self._beam_plugin_path)

        self._beam_original_animate = None
        self._beam_attached = False
        self._beam_constraint = None
        self._beam_planner = None
        self._beam_adapter = None
        self._beam_dofs = None

        self._beam_rl_step = 0
        self._beam_substep = 0
        self._beam_physics_index = 0
        self._beam_audit_count = 0
        self._beam_audit_skip_count = 0
        self._beam_safe_free_substeps = 0
        self._beam_active_substeps = 0
        self._beam_near_substeps = 0
        self._beam_unilateral_rows_built = 0
        self._beam_q_free_penetration_substeps = 0
        self._beam_worst_free_clearance_m = float("inf")
        self._beam_worst_audited_clearance_m = float("inf")
        self._beam_last_substep_record: dict[str, Any] | None = None
        self._beam_last_substep_audited = False

        super().__init__(*args, **kwargs)

    @property
    def beam_safety_validation_mode(self) -> str:
        return self._beam_validation.mode

    def _restore_animate(self) -> None:
        if self._beam_attached and self._beam_original_animate is not None:
            try:
                self.sofa_simulation.animate = self._beam_original_animate
            except Exception:
                pass
        self._beam_attached = False
        self._beam_original_animate = None

    def _validate_solver(self) -> None:
        solvers = [
            obj.getClassName()
            for obj in self._sofa_root_node.objects
            if obj.getClassName()
            in ("GenericConstraintSolver", "LCPConstraintSolver")
        ]
        if solvers != ["GenericConstraintSolver"]:
            raise RuntimeError(
                "Beam safety requires exactly one GenericConstraintSolver; "
                f"found {solvers}"
            )

    def _attach_beam_safety(self) -> None:
        if self._beam_attached:
            return
        self._validate_solver()

        instrument = self.mcr_controller_sofa.instrument.InstrumentCombined
        self._beam_dofs = instrument.getObject("DOFs")
        if self._beam_dofs is None:
            raise RuntimeError("Beam Rigid3 DOFs not found")

        self._beam_adapter = LiveBeamSDFAdapter(self, instrument)
        self._beam_constraint = instrument.addObject(
            "BeamLinearizedUnilateralConstraint",
            name="ProductionBeamLinearizedUnilateralConstraint",
            enabled=False,
            rowOffsets=[],
            dofIndices=[],
            linearJacobian=[],
            angularJacobian=[],
            freeViolations=[],
            sourceClearances=[],
        )
        self._beam_constraint.init()

        self._beam_planner = FastBeamUnilateralPlanner(
            name="ProductionFastBeamUnilateralPlanner",
            beam_dofs=self._beam_dofs,
            adapter=self._beam_adapter,
            constraint=self._beam_constraint,
            requested_margin_m=MARGIN_M,
        )
        instrument.addObject(self._beam_planner)

        self._beam_original_animate = self.sofa_simulation.animate
        self.sofa_simulation.animate = self._beam_traced_animate
        self._beam_attached = True

    def _audit_committed(self, reason: str) -> float:
        q = _as_array(self._beam_dofs.position)
        if not np.isfinite(q).all():
            raise RuntimeError(
                f"BEAM_SAFETY_NON_FINITE_COMMITTED_STATE:{reason}"
            )
        measured = self._beam_adapter.measure(q, spacing_m=DENSE_SPACING_M)
        clearance = float(measured["min_clearance_m"])
        if not np.isfinite(clearance):
            raise RuntimeError(
                f"BEAM_SAFETY_NON_FINITE_COMMITTED_CLEARANCE:{reason}"
            )
        self._beam_audit_count += 1
        self._beam_worst_audited_clearance_m = min(
            self._beam_worst_audited_clearance_m,
            clearance,
        )
        if clearance < -float(
            self._beam_validation.committed_penetration_limit_m
        ):
            raise RuntimeError(
                "BEAM_SAFETY_COMMITTED_PENETRATION:"
                f"reason={reason},clearance_mm={clearance * 1000.0:.9f}"
            )
        return clearance

    def _should_audit_substep(self, record: dict[str, Any]) -> tuple[bool, str]:
        mode = self._beam_validation.mode
        if mode == "debug":
            return True, "debug_every_substep"
        if mode == "off":
            return False, "off"

        if bool(record.get("rows_required", False)):
            return True, "active_rows"

        free_clearance = float(record.get("q_free_clearance_m", np.nan))
        if (
            np.isfinite(free_clearance)
            and free_clearance
            <= float(self._beam_validation.audit_near_clearance_m)
        ):
            return True, "near_wall"

        if (
            self._beam_physics_index
            % int(self._beam_validation.audit_interval_substeps)
            == 0
        ):
            return True, "periodic"
        return False, "skip_far_safe_free"

    def _beam_traced_animate(self, root, dt):
        self._beam_substep += 1
        self._beam_physics_index += 1
        self._beam_planner.current_rl_step = int(self._beam_rl_step)
        self._beam_planner.current_substep = int(self._beam_substep)

        result = self._beam_original_animate(root, dt)
        record = self._beam_planner.take_record()
        if record is None:
            raise RuntimeError(
                "BEAM_SAFETY_COLLISION_BEGIN_EVENT_NOT_OBSERVED"
            )
        if str(record.get("planning_status")) == "FAIL":
            raise RuntimeError(
                "BEAM_SAFETY_PLANNER_FAILED:"
                + str(record.get("reason", "unknown"))
            )

        try:
            active_count = int(self._beam_constraint.activeCount.value)
        except Exception:
            active_count = -1
        expected_rows = int(record.get("row_count", 0))
        rows_required = bool(record.get("rows_required", False))
        mismatch = (
            (
                expected_rows <= 0
                or active_count != expected_rows
                or not bool(record.get("rows_armed", False))
            )
            if rows_required
            else active_count not in (0, -1)
        )
        if mismatch:
            raise RuntimeError(
                "BEAM_SAFETY_ACTIVE_COUNT_MISMATCH:"
                f"expected={expected_rows},active={active_count}"
            )
        if rows_required:
            self._beam_active_substeps += 1
            self._beam_unilateral_rows_built += expected_rows
        else:
            self._beam_safe_free_substeps += 1

        free_clearance = float(record.get("q_free_clearance_m", np.nan))
        if np.isfinite(free_clearance):
            self._beam_worst_free_clearance_m = min(
                self._beam_worst_free_clearance_m,
                free_clearance,
            )
            if free_clearance < 0.0:
                self._beam_q_free_penetration_substeps += 1
        if (
            np.isfinite(free_clearance)
            and free_clearance
            <= float(self._beam_validation.audit_near_clearance_m)
        ):
            self._beam_near_substeps += 1

        q_committed = _as_array(self._beam_dofs.position)
        if not np.isfinite(q_committed).all():
            raise RuntimeError("BEAM_SAFETY_NON_FINITE_COMMITTED_STATE")

        do_audit, audit_reason = self._should_audit_substep(record)
        audited_clearance = None
        if do_audit:
            audited_clearance = self._audit_committed(audit_reason)
        else:
            self._beam_audit_skip_count += 1

        record["constraint_active_count"] = active_count
        record["committed_audited"] = bool(do_audit)
        record["committed_audit_reason"] = audit_reason
        record["committed_audited_clearance_m"] = audited_clearance
        self._beam_last_substep_record = record
        self._beam_last_substep_audited = bool(do_audit)
        return result

    def _safety_info(self) -> dict[str, Any]:
        worst = self._beam_worst_audited_clearance_m
        return {
            "mode": self._beam_validation.mode,
            "audit_interval_substeps": int(
                self._beam_validation.audit_interval_substeps
            ),
            "audit_near_clearance_m": float(
                self._beam_validation.audit_near_clearance_m
            ),
            "requested_margin_m": MARGIN_M,
            "dense_spacing_m": DENSE_SPACING_M,
            "committed_penetration_limit_m": float(
                self._beam_validation.committed_penetration_limit_m
            ),
            "physics_substeps_seen": int(self._beam_physics_index),
            "safe_free_substeps": int(self._beam_safe_free_substeps),
            "active_substeps": int(self._beam_active_substeps),
            "near_wall_substeps": int(self._beam_near_substeps),
            "unilateral_rows_built": int(self._beam_unilateral_rows_built),
            "q_free_penetration_substeps": int(
                self._beam_q_free_penetration_substeps
            ),
            "worst_free_clearance_m": (
                float(self._beam_worst_free_clearance_m)
                if np.isfinite(self._beam_worst_free_clearance_m)
                else None
            ),
            "committed_audits": int(self._beam_audit_count),
            "committed_audit_skips": int(self._beam_audit_skip_count),
            "worst_audited_committed_clearance_m": (
                float(worst) if np.isfinite(worst) else None
            ),
            "plugin": str(self._beam_plugin_path),
        }

    def reset(self, *args, **kwargs):
        # MCREnv/SofaEnv rebuilds the SOFA scene on reset. Restore the old
        # module animate first, then attach fresh objects to the new scene.
        self._restore_animate()
        observation, info = super().reset(*args, **kwargs)
        self._attach_beam_safety()

        self._beam_rl_step = 0
        self._beam_substep = 0
        self._beam_physics_index = 0
        self._beam_audit_count = 0
        self._beam_audit_skip_count = 0
        self._beam_safe_free_substeps = 0
        self._beam_active_substeps = 0
        self._beam_near_substeps = 0
        self._beam_unilateral_rows_built = 0
        self._beam_q_free_penetration_substeps = 0
        self._beam_worst_free_clearance_m = float("inf")
        self._beam_worst_audited_clearance_m = float("inf")
        self._beam_last_substep_record = None
        self._beam_last_substep_audited = False
        self._beam_planner.reset()

        if self._beam_validation.mode != "off":
            self._audit_committed("reset")
        if not isinstance(info, dict):
            info = {}
        info = dict(info)
        info["beam_safety"] = self._safety_info()
        return observation, info

    def step(self, action):
        self._beam_rl_step += 1
        self._beam_substep = 0
        self._beam_last_substep_audited = False

        observation, reward, terminated, truncated, info = super().step(action)

        # In audit mode, guarantee a terminal-state audit even if the final
        # physics substep happened to be a far-safe periodic skip.
        if (
            self._beam_validation.mode == "audit"
            and (terminated or truncated)
            and not self._beam_last_substep_audited
        ):
            clearance = self._audit_committed("terminal")
            if self._beam_last_substep_record is not None:
                self._beam_last_substep_record[
                    "terminal_committed_audited_clearance_m"
                ] = clearance

        if not isinstance(info, dict):
            info = {}
        info = dict(info)
        info["beam_safety"] = self._safety_info()
        return observation, reward, terminated, truncated, info

    def close(self):
        try:
            self._restore_animate()
        finally:
            return super().close()


__all__ = [
    "BeamSafetyMCREnv",
    "BeamSafetyValidationConfig",
    "DEFAULT_COMMITTED_PENETRATION_LIMIT_M",
]
