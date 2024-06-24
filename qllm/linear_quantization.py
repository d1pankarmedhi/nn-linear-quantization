from typing import List

import torch
import torch.nn as nn

from qllm.quantizer import Quantizer


class LinearTensorQuantizer(Quantizer):
    def __init__(self, tensor, scale, zero_point, dtype=torch.int8):
        self.tensor = tensor
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype

    @classmethod
    def get_symmetric_scale(cls, tensor, dtype=torch.int8):
        r_max = tensor.abs().max().item()
        q_max = torch.iinfo(dtype).max
        return r_max / q_max

    def quantize(self, *args, **kwargs):
        """
        Performs simple linear quantization given
        the scale and zero-point.
        """
        scaled_and_shifted_tensor = self.tensor / self.scale + self.zero_point
        rounded_tensor = torch.round(scaled_and_shifted_tensor)
        q_min, q_max = torch.iinfo(self.dtype).min, torch.iinfo(self.dtype).max
        q_tensor = rounded_tensor.clamp(q_min, q_max).to(self.dtype)
        return q_tensor

    def dequantize(self, quantized_tensor, *args, **kwargs):
        """
        Linear de-quantization
        """
        dequantized_tensor = self.scale * (quantized_tensor.float() - self.zero_point)
        return dequantized_tensor


class LinearQuantizer(Quantizer):
    def __init__(self, modules_to_exclude: List[str]) -> None:
        super().__init__()

    @classmethod
    def replace_modules(cls, model, target_layer, module_name_to_exclude):
        for name, child in model.named_children():
            if isinstance(child, nn.Linear) and not any(
                [x == name for x in module_name_to_exclude]
            ):
                old_bias = child.bias

                new_module = target_layer(
                    child.in_features,
                    child.out_features,
                    old_bias is not None,
                    child.weight.dtype,
                )
                setattr(model, name, new_module)
                if old_bias is not None:
                    getattr(model, name).bias = old_bias
            else:
                # Recursively call the function for nested modules
                cls.replace_modules(child, target_layer, module_name_to_exclude)

    @classmethod
    def replace_and_quantize_modules(cls, model, target_layer, module_name_to_exclude):
        for name, child in model.named_children():
            if isinstance(child, nn.Linear) and not any(
                [x == name for x in module_name_to_exclude]
            ):
                old_bias = child.bias
                old_weight = child.weight

                new_module = target_layer(
                    child.in_features,
                    child.out_features,
                    old_bias is not None,
                    child.weight.dtype,
                )
                setattr(model, name, new_module)

                getattr(model, name).quantize(old_weight)

                if old_bias is not None:
                    getattr(model, name).bias = old_bias
            else:
                # Recursively call the function for nested modules
                cls.replace_and_quantize_modules(
                    child, target_layer, module_name_to_exclude
                )
