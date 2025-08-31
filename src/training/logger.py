import os
import json
import csv
import torch
import torchinfo
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from collections import defaultdict
from src.utils.utils import write_audio_batch, ema
import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.prop_cycle': plt.cycler(color=plt.cm.plasma(np.linspace(0, 1, 5)))
})


class Logger:
    def __init__(
            self,
            base_dir: str = '../checkpoints',
            log_interval: int = 100,
            checkpoint_interval: int = 1000,
            config: Optional[Dict[str, Any]] = None
    ):
        # Setup experiment directory
        exp_name = f"{config['model']['model_type'].upper()}_{config['metadata']['dataset_used']}"
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.exp_dir = os.path.join(base_dir, f"{self.timestamp}_{exp_name}")
        os.makedirs(self.exp_dir, exist_ok=True)

        # Core tracking
        self.config = config
        self.step = 0
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval

        # Metrics storage (step-based for training, epoch-based for validation)
        self.train_metrics = defaultdict(list)  # [(step, value), ...]
        self.val_metrics = []  # [{'epoch': 1, 'loss': 0.5, ...}, ...]
        self.learning_rates = []  # [(step, lr), ...]

        # Best model tracking
        self.best_loss = float('inf')
        self.best_epoch = 0

        # Directory structure
        self.best_dir = os.path.join(self.exp_dir, 'best')
        self.plots_dir = os.path.join(self.exp_dir, 'plots')
        os.makedirs(self.best_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        # Save config immediately
        if config:
            self.save_config(config)

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save experiment configuration."""
        self.config = config
        with open(os.path.join(self.exp_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def log_model_summary(self, model: torch.nn.Module, dataloader) -> None:
        """Generate and save model summary."""
        input_size = [
            (1, *dataloader.dataset.inputs.shape[1:]),
            (1, *dataloader.dataset.controls.shape[1:])
        ]

        summary = torchinfo.summary(
            model,
            input_size=input_size,
            col_names=["input_size", "output_size", "num_params"],
            verbose=0
        )

        with open(os.path.join(self.exp_dir, 'model_summary.txt'), 'w') as f:
            f.write(str(summary))

    def log_step(self, metrics: Dict[str, float], learning_rate: float) -> None:
        """Log training step metrics."""
        self.step += 1

        # Store metrics
        for name, value in metrics.items():
            self.train_metrics[name].append((self.step, value))

        self.learning_rates.append((self.step, learning_rate))

    def log_epoch(self, epoch: int, val_metrics: Dict[str, float]) -> bool:
        """Log validation metrics for epoch and return if this is best model."""
        val_record = {'epoch': epoch, **val_metrics}
        self.val_metrics.append(val_record)

        # Check and update best model
        current_loss = val_metrics.get('total_loss', float('inf'))
        is_best = current_loss < self.best_loss

        if is_best:
            self.best_loss = current_loss
            self.best_epoch = epoch

        return is_best

    def save_plots(self) -> None:
        """Create and save loss plots."""
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16, 10), sharex=True)

        # Training losses
        for name, data in self.train_metrics.items():
            if data:
                steps, values = zip(*data)
                ax1.plot(steps, values, label=name, alpha=0.7)

                # Add smoothed line for total_loss
                if name == 'total_loss':
                    ax1.plot(steps, ema(values), color='teal', linewidth=2,
                             label='total_loss (smoothed)')

        ax1.set_ylabel('Loss')
        ax1.set_title('Training Losses')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Learning rate
        if self.learning_rates:
            steps, lrs = zip(*self.learning_rates)
            ax2.plot(steps, lrs, color='teal', label='Learning Rate')

        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, 'training_curves.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    def save_metrics_csv(self) -> None:
        """Save metrics to CSV files - only called at end of training."""
        # Training loss CSV
        train_path = os.path.join(self.exp_dir, 'train_loss.csv')

        # Pre-allocate and avoid repeated dict operations
        if not self.train_metrics:
            return

        # Get all steps and metrics efficiently
        all_steps = set()
        for data in self.train_metrics.values():
            all_steps.update(step for step, _ in data)

        if not all_steps:
            return

        all_steps = sorted(all_steps)
        metric_names = sorted(self.train_metrics.keys())

        # Create lookup dictionaries for O(1) access
        metric_lookups = {}
        for name, data in self.train_metrics.items():
            metric_lookups[name] = dict(data)

        lr_lookup = dict(self.learning_rates)

        # Write training CSV
        with open(train_path, 'w', newline='') as f:
            fieldnames = ['step', 'learning_rate'] + metric_names
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for step in all_steps:
                row = {'step': step, 'learning_rate': lr_lookup.get(step, '')}
                for metric in metric_names:
                    row[metric] = metric_lookups[metric].get(step, '')
                writer.writerow(row)

        # Validation loss CSV
        if self.val_metrics:
            val_path = os.path.join(self.exp_dir, 'val_loss.csv')
            with open(val_path, 'w', newline='') as f:
                fieldnames = ['epoch'] + [k for k in self.val_metrics[0].keys() if k != 'epoch']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.val_metrics)

    def save_checkpoint(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            epoch: int,
            is_best: bool = False,
            predictions: Optional[torch.Tensor] = None,
            targets: Optional[torch.Tensor] = None,
            sample_rate: int = 48000,
            save_all_epochs: bool = True,
    ) -> None:
        """Save model checkpoint."""

        checkpoint = {
            "epoch": epoch,
            "step": self.step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }

        # Save best
        if is_best:
            torch.save(checkpoint, os.path.join(self.best_dir, "checkpoint.pth"))
            torch.save(model.state_dict(), os.path.join(self.best_dir, "model.pth"))
            print(f"✓ New best model saved! Loss: {self.best_loss:.6f} at epoch {epoch + 1}")

        # Save audio samples every epoch
        if predictions is not None and targets is not None:
            epoch_audio_dir = os.path.join(self.exp_dir, 'audio_by_epoch', f'epoch_{epoch + 1:03d}')
            self.save_audio_samples(predictions, targets, sample_rate, epoch_audio_dir)

        # Save all epochs
        if save_all_epochs:
            torch.save(checkpoint, os.path.join(epoch_audio_dir, "checkpoint.pth"))
            torch.save(model.state_dict(), os.path.join(epoch_audio_dir, "model.pth"))

        # Save last model
        last_dir = os.path.join(self.exp_dir, 'last')
        os.makedirs(last_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(last_dir, "checkpoint.pth"))

    @staticmethod
    def save_audio_samples(
            predictions: torch.Tensor,
            targets: torch.Tensor,
            sample_rate: int,
            save_dir: str
    ) -> None:
        """Save audio samples."""
        audio_dir = os.path.join(save_dir, 'audio')
        os.makedirs(audio_dir, exist_ok=True)

        pred_dir = os.path.join(audio_dir, 'predictions')
        target_dir = os.path.join(audio_dir, 'targets')
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)

        write_audio_batch(pred_dir, predictions, sample_rate)
        write_audio_batch(target_dir, targets, sample_rate)

    def finalize(self) -> Dict[str, Any]:
        """Finalize logging and create summary."""
        # Save all artifacts
        self.save_plots()
        self.save_metrics_csv()

        # Create clean summary
        summary = {
            'experiment_id': os.path.basename(self.exp_dir),
            'total_steps': self.step,
            'total_epochs': len(self.val_metrics),
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.train_metrics['total_loss'][-1][1] if self.train_metrics['total_loss'] else None,
            'final_val_loss': self.val_metrics[-1]['total_loss'] if self.val_metrics else None,
            'experiment_dir': self.exp_dir,
            'completed_at': datetime.now().isoformat()
        }

        # Save summary
        with open(os.path.join(self.exp_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 50}")
        print(f"Training completed!")
        print(f"Experiment: {summary['experiment_id']}")
        print(f"Best loss: {summary['best_loss']:.6f} at epoch {summary['best_epoch']}")
        print(f"Results: {self.exp_dir}")
        print(f"{'=' * 50}")

        return summary


def setup_logger(config: Dict[str, Any]) -> Logger:
    """Initialize logger from config."""
    logger_cfg = config['logger']
    return Logger(
        base_dir=logger_cfg['checkpoints_dir'],
        checkpoint_interval=logger_cfg['checkpoint_interval'],
        config=config
    )
