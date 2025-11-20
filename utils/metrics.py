
import torch

class MeanMetric:
    """Keras-like streaming mean for PyTorch."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = torch.tensor(0.0)
        self._count = torch.tensor(0.0)

    def update(self, value, weight=1.0):
        """
        value: float, Tensor (scalar), numpy, etc.
        weight: float or int, default 1.0
        """
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(float(weight))

        # 保证同一设备
        value, weight = value.to(self._sum), weight.to(self._sum)
        self._sum += value * weight
        self._count += weight

    def result(self):
        return (self._sum / self._count).item() if self._count > 0 else 0.0

    def __call__(self, value, weight=1.0):
        """允许直接当函数用：metric(loss)"""
        self.update(value, weight)

