"""CNN model architectures for CIFAR-10 classification."""

from .cnn_baseline import build_baseline_cnn
from .cnn_advanced import build_advanced_cnn

__all__ = [
    'build_baseline_cnn',
    'build_advanced_cnn'
]
