"""Synchronized episode-level mastery checks for the existing five-stage task."""

from __future__ import annotations

from collections import deque

from mcr_sim.training_config import (
    TRAINING_CURRICULUM_CONSECUTIVE_SUCCESS_EPISODES,
    TRAINING_CURRICULUM_MIN_EPISODES_PER_VESSEL,
    TRAINING_CURRICULUM_MODELS,
    TRAINING_CURRICULUM_PROMOTION_MODES,
    TRAINING_CURRICULUM_ROLLING_EPISODES_PER_VESSEL,
    TRAINING_CURRICULUM_STAGE_NAMES,
    TRAINING_CURRICULUM_SUCCESS_THRESHOLDS,
)


class CurriculumController:
    def __init__(self, stage=0):
        self.stage = int(stage)
        if not 0 <= self.stage < len(TRAINING_CURRICULUM_MODELS):
            raise ValueError("Unknown curriculum stage")
        self._reset_window()

    def _reset_window(self):
        self.rolling = {
            model: deque(maxlen=TRAINING_CURRICULUM_ROLLING_EPISODES_PER_VESSEL)
            for model in TRAINING_CURRICULUM_MODELS[self.stage]
        }
        self.streak = 0

    def status(self):
        counts = {name: len(values) for name, values in self.rolling.items()}
        rates = {name: (sum(values) / len(values) if values else None)
                 for name, values in self.rolling.items()}
        ready = all(count >= TRAINING_CURRICULUM_MIN_EPISODES_PER_VESSEL
                    for count in counts.values())
        score = None
        if ready and self.stage < len(TRAINING_CURRICULUM_SUCCESS_THRESHOLDS):
            if TRAINING_CURRICULUM_PROMOTION_MODES[self.stage] == "aggregate":
                score = sum(sum(values) for values in self.rolling.values()) / sum(counts.values())
            else:
                score = min(float(value) for value in rates.values())
        return {
            "stage": self.stage,
            "stage_name": TRAINING_CURRICULUM_STAGE_NAMES[self.stage],
            "models": list(TRAINING_CURRICULUM_MODELS[self.stage]),
            "counts": counts, "success_rates": rates,
            "mastery_score": score,
            "threshold": (TRAINING_CURRICULUM_SUCCESS_THRESHOLDS[self.stage]
                          if self.stage < len(TRAINING_CURRICULUM_SUCCESS_THRESHOLDS) else None),
            "streak": self.streak,
            "streak_required": TRAINING_CURRICULUM_CONSECUTIVE_SUCCESS_EPISODES,
        }

    def observe(self, events):
        """Consume the same rank-ordered events on every rank; return transition(s)."""
        transitions = []
        for event in events:
            if int(event["stage"]) != self.stage:
                continue  # An episode started before promotion; do not count twice.
            model = str(event["vessel_id"])
            if model not in self.rolling:
                continue
            self.rolling[model].append(int(bool(event["success"])))
            state = self.status()
            if state["threshold"] is None or state["mastery_score"] is None:
                self.streak = 0
                continue
            self.streak = (self.streak + 1 if state["mastery_score"] >= state["threshold"] else 0)
            if self.streak >= TRAINING_CURRICULUM_CONSECUTIVE_SUCCESS_EPISODES:
                previous = state
                self.stage += 1
                self._reset_window()
                transitions.append({"from": previous, "to": self.status()})
        return transitions
