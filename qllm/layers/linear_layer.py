import torch
import torch.nn as nn
import torch.nn.functional as F


class W8A16LL(nn.Module):
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
                127,
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

    def quantize(self, weights):
        w_fp32 = weights.clone().to(torch.float32)
        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)
        int8_weights = torch.round(weights / scales.unsqueeze(1)).to(torch.int8)
        self.int8_weights = int8_weights
        self.scales = scales

    def forward(self, input):
        casted_weights = self.int8_weights.to(input.dtype)
        output = F.linear(input, casted_weights) * self.scales
        if self.bias is not None:
            output = output + self.bias
        return output
