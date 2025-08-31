import torch
from tqdm import tqdm
from typing import Dict, Any


def train_epoch(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        logger: Any,
        device: torch.device,
        pbar: tqdm,
        gradient_clip_norm: float = 5.0) -> Dict[str, float]:
    """
    Training epoch.
    """
    model.train()
    epoch_metrics = {"total_loss": 0, "harmonic_loss": 0, "stft_loss": 0}
    num_batches = 0

    for batch_idx, (inputs, control, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        control = control.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(inputs, control)
        loss, loss_terms = criterion(predictions, targets, control[:, 0, :].unsqueeze(1))

        # Check for NaN
        if torch.isnan(loss):
            raise RuntimeError(f'NaN loss detected at batch {batch_idx}')

        # Backward pass
        loss.backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        scheduler.step()

        # Metrics
        batch_metrics = {
            "total_loss": loss.item(),
            "harmonic_loss": loss_terms['harmonic'],
            "stft_loss": loss_terms['stft']
        }

        # Log step
        logger.log_step(batch_metrics, optimizer.param_groups[0]['lr'])

        # Update progress bar every step
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        pbar.update(1)

        # Accumulate epoch metrics
        for key, value in batch_metrics.items():
            epoch_metrics[key] += value
        num_batches += 1

    # Average over batches
    return {k: v / num_batches for k, v in epoch_metrics.items()}


def validate_epoch(
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: torch.device) -> Dict[str, float]:
    """
    Validation epoch.
    """
    model.eval()
    val_metrics = {"total_loss": 0, "harmonic_loss": 0, "stft_loss": 0}
    num_batches = 0

    with torch.no_grad():
        for inputs, control, targets in val_loader:
            inputs = inputs.to(device)
            control = control.to(device)
            targets = targets.to(device)

            predictions = model(inputs, control)
            loss, loss_terms = criterion(predictions, targets, control[:, 0, :].unsqueeze(1))

            val_metrics["total_loss"] += loss.item()
            val_metrics["harmonic_loss"] += loss_terms['harmonic']
            val_metrics["stft_loss"] += loss_terms['stft']
            num_batches += 1

    # Average over batches
    return {k: v / num_batches for k, v in val_metrics.items()}


def train_model(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        logger: Any,
        device: torch.device,
        epochs: int,
        gradient_clip_norm: float = 5.0) -> Dict[str, Any]:
    """
    Main training loop.
    """

    # Initialize
    logger.log_model_summary(model, train_loader)

    # Progress tracking - step level updates
    total_batches = len(train_loader) * epochs
    pbar = tqdm(total=total_batches, desc="Training", unit="step")

    try:
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Training phase
            train_metrics = train_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                logger=logger,
                device=device,
                pbar=pbar,
                gradient_clip_norm=gradient_clip_norm
            )

            # Validation phase
            val_metrics = validate_epoch(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device
            )

            # Log epoch results and check if best
            is_best = logger.log_epoch(epoch, val_metrics)

            # Get validation predictions for audio saving
            model.eval()
            with torch.no_grad():
                val_inputs, val_control, val_targets = next(iter(val_loader))
                val_inputs = val_inputs.to(device)
                val_control = val_control.to(device)
                val_targets = val_targets.to(device)
                val_predictions = model(val_inputs, val_control)

            # Save checkpoint
            logger.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                is_best=is_best,
                predictions=val_predictions,
                targets=val_targets,
                sample_rate=train_loader.dataset.sample_rate
            )

            # Write CSVs
            logger.save_metrics_csv()

            # Print epoch summary
            print(f"Train: {train_metrics['total_loss']:.4f} | "
                  f"Val: {val_metrics['total_loss']:.4f} | "
                  f"Best: {logger.best_loss:.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise
    finally:
        pbar.close()

    # Finalize logging
    summary = logger.finalize()
    return summary


def create_optimizer(config: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    cfg = config['trainer']
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get('weight_decay', 0.01),
        betas=cfg.get('betas', (0.9, 0.999))
    )


def create_scheduler(
        config: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create scheduler from config."""
    cfg = config['trainer']
    total_steps = len(dataloader) * cfg['epochs']

    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.get('max_lr', cfg['learning_rate'] * 10),
        total_steps=total_steps,
        pct_start=cfg.get('pct_start', 0.3),
        div_factor=cfg.get('div_factor', 25),
        final_div_factor=cfg.get('final_div_factor', 1e4),
        anneal_strategy=cfg.get('anneal_strategy', 'cos')
    )


def setup_logger(config: Dict[str, Any]):
    """Create logger from config."""
    from logger import Logger

    cfg = config.get('logger', {})
    return Logger(
        base_dir=cfg.get('checkpoints_dir', '../checkpoints'),
        log_interval=cfg.get('log_interval', 100),
        checkpoint_interval=cfg.get('checkpoint_interval', 1000),
        config=config
    )


def create_optimizer(config: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    """Initialize optimizer from config."""
    trainer_cfg = config['trainer']
    return torch.optim.AdamW(
        model.parameters(),
        lr=trainer_cfg["learning_rate"],
        weight_decay=trainer_cfg['weight_decay'],
        betas=trainer_cfg['betas']
    )


def create_scheduler(
        config: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        dataloader: object
) -> torch.optim.lr_scheduler.LRScheduler:
    """Initialize scheduler from config."""
    trainer_cfg = config['trainer']
    total_steps = len(dataloader) * trainer_cfg['epochs']

    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=trainer_cfg['max_lr'],
        total_steps=total_steps,
        pct_start=trainer_cfg['pct_start'],
        div_factor=trainer_cfg['div_factor'],
        final_div_factor=trainer_cfg['final_div_factor'],
        anneal_strategy=trainer_cfg['anneal_strategy']
    )
