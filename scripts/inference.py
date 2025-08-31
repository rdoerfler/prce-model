from configs.base_config import load_config
from src.data.dataloader import create_dataloader
from src.models.model import create_model
from src.utils.utils import setup_device, write_audio_batch, post_processing
import os
import torch

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset"))


def test_model(model, dataloader, steps, path, device):

    outputs = []
    targets = []

    model.eval()
    with torch.no_grad():
        for i in range(*steps):
            print(f'Example {i:02d} | Infering at i {i:02d}')
            inputs, control, target = dataloader.dataset[i]
            inputs = inputs[None, ...].to(device)
            control = control[None, ...].to(device)
            target = target[None, ...].to(device)

            # Generate Audio Prediction
            output = model(inputs, control)

            # Optional Post Processing for inspection
            output = post_processing(output, fades=True, norm=True)
            target = post_processing(target, fades=True, norm=True)

            outputs.append(output)
            targets.append(target)

        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)

        write_audio_batch(
            path=f'{path}/test_inference/predictions',
            data=outputs,
            sample_rate=dataloader.dataset.sample_rate
        )
        write_audio_batch(
            path=f'{path}/test_inference/targets',
            data=targets,
            sample_rate=dataloader.dataset.sample_rate
        )


def main():
    """Model loading and inference."""
    # Number of Examples
    num_steps = (0, 43)

    # Define Checkpoint
    experiment_name = '2025-08-26_20-32-43_HPN_A_full_set'
    experiment_path = os.path.join('checkpoints', experiment_name)

    # Load config and checkpoint
    config_path = os.path.join(experiment_path, 'config.json')
    config = load_config(config_path)

    # Update dataset for inference
    config['trainer']['batch_size'] = 1
    config['dataset']['dataset_path'] = os.path.join(DATA_DIR, config['metadata']['dataset_used'])

    # Setup components
    device = setup_device()
    model = create_model(config, device)
    dataloader = create_dataloader(config, split='test')

    # Load checkpoint weights
    checkpoint_path = os.path.join(experiment_path, 'best', 'checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Model loaded from checkpoint at epoch {checkpoint['epoch']}")
    print(f"Best loss: {checkpoint.get('best_loss', 'N/A')}")

    # Run training
    test_model(
        model=model,
        dataloader=dataloader,
        path=experiment_path,
        steps=num_steps,
        device=device
    )


if __name__ == '__main__':
    main()
