from src.cli import cli
from configs.base_config import get_config, update_config
from src.data.dataloader import create_dataloader
from src.models.model import create_model
from src.training.loss import create_loss_fn
from src.training.logger import setup_logger
from src.training.trainer import train_model, create_optimizer, create_scheduler
from src.utils.utils import setup_device


def main():
    """Main training setup and execution."""
    args = cli()
    config = get_config()
    config = update_config(config, args)

    # Setup components
    device = setup_device()
    logger = setup_logger(config)
    model = create_model(config, device)
    loss_fn = create_loss_fn(config, device)
    optimizer = create_optimizer(config, model)

    # Train Dataloader
    train_dataloader = create_dataloader(config, 'train')

    # Save the complete config (with updated mean, std)
    logger.save_config(config)

    # Val Dataloader with train mean, std from config
    val_dataloader = create_dataloader(config, 'val')

    scheduler = create_scheduler(config, optimizer, train_dataloader)

    # Run training
    summary = train_model(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        criterion=loss_fn,
        optimizer=optimizer,
        epochs=config['trainer']['epochs'],
        scheduler=scheduler,
        logger=logger,
        device=device
    )

    return summary


if __name__ == '__main__':
    main()

