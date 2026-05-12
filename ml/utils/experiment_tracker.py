"""
MLflow + W&B Experiment Tracking Integration
Unified tracking interface for all ML training runs.
"""
from __future__ import annotations
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional
import structlog

logger = structlog.get_logger(__name__)


class ExperimentTracker:
    """
    Unified experiment tracking facade supporting MLflow and Weights & Biases.
    Gracefully handles unavailable backends.
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        use_mlflow: bool = True,
        use_wandb: bool = True,
        tags: Optional[Dict[str, str]] = None,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{int(time.time())}"
        self.tags = tags or {}
        self._mlflow_run = None
        self._wandb_run = None
        self._active = False

        self._init_mlflow(use_mlflow)
        self._init_wandb(use_wandb)

    def _init_mlflow(self, enabled: bool):
        if not enabled:
            return
        try:
            import mlflow
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
            logger.info("mlflow_initialized", uri=tracking_uri, experiment=self.experiment_name)
        except ImportError:
            self._mlflow = None
            logger.warning("mlflow_not_available")

    def _init_wandb(self, enabled: bool):
        if not enabled:
            self._wandb = None
            return
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            self._wandb = None
            logger.warning("wandb_not_available")

    def start(self):
        """Start experiment tracking run."""
        if self._mlflow:
            self._mlflow_run = self._mlflow.start_run(run_name=self.run_name, tags=self.tags)
        if self._wandb:
            try:
                project = os.environ.get("WANDB_PROJECT", "panoramic-360-engine")
                entity = os.environ.get("WANDB_ENTITY")
                self._wandb_run = self._wandb.init(
                    project=project, entity=entity,
                    name=self.run_name, tags=list(self.tags.values()),
                    config=self.tags, reinit=True,
                )
            except Exception as e:
                logger.warning("wandb_init_failed", error=str(e))
        self._active = True
        logger.info("experiment_started", run=self.run_name)

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if self._mlflow and self._mlflow_run:
            self._mlflow.log_params(params)
        if self._wandb_run:
            self._wandb_run.config.update(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log scalar metrics."""
        if self._mlflow and self._mlflow_run:
            self._mlflow.log_metrics(metrics, step=step)
        if self._wandb_run:
            log_data = {**metrics}
            if step is not None:
                log_data["_step"] = step
            self._wandb_run.log(log_data, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a file artifact."""
        if self._mlflow and self._mlflow_run:
            self._mlflow.log_artifact(local_path, artifact_path)
        if self._wandb_run:
            self._wandb_run.save(local_path)

    def log_image(self, key: str, image, step: Optional[int] = None):
        """Log an image (numpy array or PIL Image)."""
        if self._wandb_run:
            try:
                self._wandb_run.log({key: self._wandb.Image(image)}, step=step)
            except Exception:
                pass

    def set_tags(self, tags: Dict[str, str]):
        """Set run tags."""
        if self._mlflow and self._mlflow_run:
            for k, v in tags.items():
                self._mlflow.set_tag(k, v)
        self.tags.update(tags)

    def finish(self, status: str = "success"):
        """Finalize and close the run."""
        if self._mlflow and self._mlflow_run:
            self._mlflow.set_tag("status", status)
            self._mlflow.end_run()
        if self._wandb_run:
            self._wandb_run.finish(exit_code=0 if status == "success" else 1)
        self._active = False
        logger.info("experiment_finished", run=self.run_name, status=status)

    @contextmanager
    def run(self):
        """Context manager for automatic start/finish."""
        self.start()
        try:
            yield self
            self.finish("success")
        except Exception as e:
            self.finish("failed")
            logger.error("experiment_failed", error=str(e))
            raise

    def log_segmentation_metrics(self, metrics: Dict[str, float], epoch: int):
        """Convenience method for segmentation-specific metrics."""
        prefixed = {f"seg/{k}": v for k, v in metrics.items()}
        self.log_metrics(prefixed, step=epoch)

    def log_ppe_metrics(self, metrics: Dict[str, float], epoch: int):
        """Convenience method for PPE-specific metrics."""
        prefixed = {f"ppe/{k}": v for k, v in metrics.items()}
        self.log_metrics(prefixed, step=epoch)


class EarlyStopping:
    """Early stopping with patience and minimum delta."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.wait = 0
        self.stopped = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        improved = (
            score >= self.best_score + self.min_delta if self.mode == "max"
            else score <= self.best_score - self.min_delta
        )
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped = True
                logger.info("early_stopping_triggered",
                    best_epoch=self.best_epoch,
                    best_score=f"{self.best_score:.4f}",
                    waited=self.wait)
                return True
        return False
