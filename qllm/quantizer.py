from abc import ABC, abstractmethod


class Quantizer(ABC):
    @abstractmethod
    def quantize(self, *args, **kwargs):
        pass

    @abstractmethod
    def dequantize(self, *args, **kwargs):
        pass
