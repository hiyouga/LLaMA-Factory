#!/usr/bin/env python3

"""Working custom callbacks for real LLaMA-Factory training.

These can be used directly with YAML configuration files.
"""

import time
from typing import TYPE_CHECKING, Optional

from transformers import TrainerCallback, TrainerControl, TrainerState


if TYPE_CHECKING:
    from transformers import TrainingArguments
from src.llamafactory.extras import logging


logger = logging.get_logger("llamafactory.callbacks.working_custom_callbacks")


class DemoUploadMonitorCallBack(TrainerCallback):
    """Real working callback that monitors training and uploads metrics."""

    """This demonstrates a practical use case for custom callbacks."""

    def __init__(
        self,
        upload_url: str = "https://demo-monitor.internal/api/metrics",
        api_key: str = "demo_key",
        project_name: str = "llama-factory-training",
        upload_interval: int = 50,
    ):
        self.upload_url = upload_url
        self.api_key = api_key
        self.project_name = project_name
        self.upload_interval = upload_interval
        self.step_count = 0
        self.start_time = None

        logger.info("🔧 DemoUploadMonitorCallBack initialized:")
        logger.info(f"   Project: {self.project_name}")
        logger.info(f"   Upload URL: {self.upload_url}")
        logger.info(f"   Upload interval: {self.upload_interval} steps")

    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called when training starts."""
        self.start_time = time.time()
        logger.info(f"🚀 [Demo Upload] Training started for project: {self.project_name}")
        logger.info(f"   Model: {getattr(args, 'output_dir', 'Unknown')}")
        logger.info(f"   Total epochs: {args.num_train_epochs}")
        logger.debug("[DEBUG] on_train_begin called in DemoUploadMonitorCallBack")

    def on_log(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict[str, float]] = None,
        **kwargs,
    ):
        """Called when metrics are logged."""
        logger.debug(f"[DEBUG] on_log called in DemoUploadMonitorCallBack, logs={logs}")
        if logs is None:
            return

        self.step_count += 1

        # Upload metrics at specified intervals
        if self.step_count % self.upload_interval == 0:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            # Simulate uploading to Demo Upload monitoring system
            logger.info(f"📊 [Demo Upload] Uploading metrics at step {state.global_step}")
            logger.info(f"   Project: {self.project_name}")
            logger.info(f"   Elapsed time: {elapsed_time:.1f}s")
            logger.info(f"   Current loss: {logs.get('train_loss', 'N/A')}")
            logger.info(f"   Learning rate: {logs.get('learning_rate', 'N/A')}")

            # In a real implementation, you would make an HTTP request here:
            # import requests
            # payload = {
            #     "project": self.project_name,
            #     "step": state.global_step,
            #     "epoch": state.epoch,
            #     "metrics": logs,
            #     "timestamp": time.time()
            # }
            # requests.post(self.upload_url, json=payload,
            #               headers={"Authorization": f"Bearer {self.api_key}"})

    def on_evaluate(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict[str, float]] = None,
        **kwargs,
    ):
        """Called after evaluation."""
        if logs:
            logger.info(f"📈 [Demo Upload] Evaluation at step {state.global_step}")
            logger.info(f"   Eval loss: {logs.get('eval_loss', 'N/A')}")
            if "eval_loss" in logs and "train_loss" in logs:
                gap = logs["eval_loss"] - logs["train_loss"]
                logger.info(f"   Train/Eval gap: {gap:.4f}")

    def on_train_end(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called when training ends."""
        if self.start_time:
            total_time = time.time() - self.start_time
            logger.info(f"🏁 [Demo Upload] Training completed for project: {self.project_name}")
            logger.info(f"   Total time: {total_time:.1f}s")
            logger.info(f"   Total steps: {state.global_step}")
            logger.info(f"   Final epoch: {state.epoch:.2f}")


class SmartEarlyStoppingCallback(TrainerCallback):
    """Smart early stopping with custom logic and alerting."""

    def __init__(self, loss_threshold: float = 5.0, patience: int = 3):
        self.loss_threshold = loss_threshold
        self.patience = patience
        self.high_loss_count = 0
        self.best_eval_loss = float("inf")
        self.no_improvement_count = 0

        logger.info("🛑 SmartEarlyStoppingCallback initialized:")
        logger.info(f"   Loss threshold: {self.loss_threshold}")
        logger.info(f"   Patience: {self.patience}")

    def on_evaluate(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict[str, float]] = None,
        **kwargs,
    ):
        """Check for early stopping conditions."""
        if logs is None:
            return control

        eval_loss = logs.get("eval_loss")
        if eval_loss is None:
            return control

        logger.info(f"🔍 [SMART STOP] Checking early stopping at step {state.global_step}")
        logger.info(f"   Current eval loss: {eval_loss:.4f}")
        logger.info(f"   Best eval loss: {self.best_eval_loss:.4f}")

        # Check for improvement
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.no_improvement_count = 0
            logger.info("   ✅ New best eval loss! Reset patience counter.")
        else:
            self.no_improvement_count += 1
            logger.info(f"   ⚠️  No improvement ({self.no_improvement_count}/{self.patience})")

        # Check loss threshold
        if eval_loss > self.loss_threshold:
            self.high_loss_count += 1
            logger.warning(f"   🚨 High loss detected! Count: {self.high_loss_count}")
            if self.high_loss_count >= 2:
                logger.error(f"   🛑 Stopping: Loss {eval_loss:.4f} > threshold {self.loss_threshold}")
                control.should_training_stop = True
                return control
        else:
            self.high_loss_count = 0

        # Check patience
        if self.no_improvement_count >= self.patience:
            logger.error(f"   🛑 Stopping: No improvement for {self.patience} evaluations")
            control.should_training_stop = True

        return control


class ModelAnalysisCallback(TrainerCallback):
    """Analyzes model state during training for debugging and monitoring."""

    def __init__(self, analysis_steps: int = 100):
        self.analysis_steps = analysis_steps
        self.step_count = 0

        logger.info("🔍 ModelAnalysisCallback initialized:")
        logger.info(f"   Analysis every {self.analysis_steps} steps")

    def on_step_end(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Analyze model at specified intervals."""
        self.step_count += 1

        if self.step_count % self.analysis_steps == 0:
            model = kwargs.get("model")
            optimizer = kwargs.get("optimizer")

            logger.info(f"🔬 [MODEL ANALYSIS] Step {state.global_step}")

            if model is not None:
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                logger.info("   📊 Parameters:")
                logger.info(f"      Total: {total_params:,}")
                logger.info(f"      Trainable: {trainable_params:,} ({trainable_params/total_params:.1%})")

                # Check for NaN/inf parameters
                nan_params = sum(1 for p in model.parameters() if p.data.isnan().any())
                inf_params = sum(1 for p in model.parameters() if p.data.isinf().any())

                if nan_params > 0 or inf_params > 0:
                    logger.warning(f"   ⚠️  Warning: NaN params: {nan_params}, Inf params: {inf_params}")
                else:
                    logger.info("   ✅ All parameters healthy")

            if optimizer is not None:
                # Get current learning rate
                lr = optimizer.param_groups[0].get("lr", "N/A")
                logger.info(f"   📈 Learning rate: {lr}")
                # Check optimizer state
                logger.info(f"   🔧 Optimizer: {type(optimizer).__name__}")


# Demo callback that shows environment variable usage
class EnvironmentAwareCallback(TrainerCallback):
    """Demonstrates how to use environment variables in callback configuration."""

    def __init__(
        self,
        debug_mode: str = "false",  # Will come from ${DEBUG_MODE:-false}
        log_file: str = "./training.log",
        project_name: str = "Pet Project",
    ):
        self.debug_mode = debug_mode.lower() == "true"
        self.log_file = log_file
        self.project_name = project_name

        logger.info("🌍 EnvironmentAwareCallback initialized:")
        logger.info(f"   Project Name: {self.project_name}")
        logger.info(f"   Debug mode: {self.debug_mode}")
        logger.info(f"   Log file: {self.log_file}")

    def on_log(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict[str, float]] = None,
        **kwargs,
    ):
        """Log with environment-based configuration."""
        if logs and self.debug_mode:
            logger.debug(f"🐛 [DEBUG {self.project_name}] Detailed logging at step {state.global_step}")
            for key, value in logs.items():
                logger.debug(f"      {key}: {value}")

        # Write to log file
        if logs:
            try:
                with open(self.log_file, "a") as f:
                    f.write(f"[{self.project_name}] Step {state.global_step}: {logs}\n")
            except Exception as e:
                logger.warning(f"   ⚠️  Failed to write to log file: {e}")


if __name__ == "__main__":
    logger.info("🔧 Custom callbacks for LLaMA-Factory are ready!")
    logger.info("These callbacks can be used in YAML configuration files:")
    logger.info("")
    logger.info("custom_callbacks:")
    logger.info("  - name: 'examples.custom_callbacks.DemoUploadMonitorCallBack'")
    logger.info("    args:")
    logger.info("      project_name: 'my-experiment'")
    logger.info("      upload_interval: 50")
    logger.info("")
    logger.info("Then run: llamafactory-cli train your_config.yaml")
