import torch
import torch.nn as nn
import torch.nn.functional as F

from qllm.quantizer import Quantizer


class LinearQuantizer(Quantizer):
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


class W8A16LLqllm(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        dtype=torch.float32,
    ) -> None:
        super().__init__()

        self.register_buffer(
            "int8_weights",
            torch.randint(
                -128,
                128,
                (out_features, in_features),
                dtype=torch.int8,
            ),
        )
        self.register_buffer(
            "scales",
            torch.randn(
                (out_features),
                dtype=dtype,
            ),
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.randn(
                    (1, out_features),
                    dtype=dtype,
                ),
            )

        else:
            self.bias = None

    def w8_a16_forward(weight, input, scales, bias=None):
        casted_weights = weight.to(input.dtype)
        output = F.linear(input, casted_weights) * scales

        if bias is not None:
            output = output + bias

        return output

    def forward(self, input):
        casted_weights = self.int8_weights.to(input.dtype)
        output = F.linear(input, casted_weights) * self.scales
        if self.bias is not None:
            output = output + self.bias

        return output
