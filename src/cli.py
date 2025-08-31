import argparse

"""
Command Line Interface for ProceduralEnginesModles training and inference.
"""


def cli():
    argparser = argparse.ArgumentParser(description='Command Line Interface for ProceduralEnginesModles.')

    argparser.add_argument(
        '--model_type',
        type=str,
        default='ptr',
        help='Chose HarmonicPlusNoise (hpn) or PulseTrainResonator (ptr) model.'
    )

    argparser.add_argument(
        '--dataset',
        type=str,
        default='C_full_set',
        help='Name of dataset (or subset) for training | inference.'
    )

    argparser.add_argument(
        '--num_harmonics',
        type=int,
        default=100,
        help='Number of harmonics.'
    )

    argparser.add_argument(
        '--num_noisebands',
        type=int,
        default=256,
        help='Number of noise bands.'
    )

    argparser.add_argument(
        '--hidden_size',
        type=int,
        default=256,
        help='Size of hidden layers.'
    )

    argparser.add_argument(
        '--gru_size',
        type=int,
        default=512,
        help='Size of gru layer.'
    )

    return argparser.parse_args()
