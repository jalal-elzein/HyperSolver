import os
import time

import wandb


class RunLogger:
    """
    Thin wrapper around a single W&B run, per wandb_tracking_spec.md.

    One instance = one (instance, method, condition, seed) solve. Owns the
    monotonic step counter so it keeps counting across the pre_refinement ->
    post_refinement boundary instead of each phase resetting to 0.
    """

    PROJECT = "MS3HN2-Evaluation"

    def __init__(self):
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError(
                "WANDB_API_KEY is not set. W&B logging is required for this run; "
                "export WANDB_API_KEY before running run.py with --problem hypermaxcut."
            )
        self._run = None
        self._step = 0
        self._start_time = None

    def start(self, config: dict):
        self._run = wandb.init(project=self.PROJECT, config=config)
        self._step = 0
        self._start_time = time.time()

    def log_step(self, phase: str, quality: float, feasible=None):
        payload = {
            "phase": phase,
            "wall_time_s": time.time() - self._start_time,
            "quality": quality,
        }
        if feasible is not None:
            payload["feasible"] = feasible
        wandb.log(payload, step=self._step)
        self._step += 1

    def set_summary(self, **kwargs):
        for k, v in kwargs.items():
            self._run.summary[k] = v

    def finish(self):
        wandb.finish()