from qllm.linear_quantization import LinearQuantizer
from qllm.quantizer import (
    replace_linear_with_target,
    replace_linear_with_target_and_quantize,
)

__all__ = [
    "LinearQuantizer",
    "replace_linear_with_target",
    "replace_linear_with_target_and_quantize",
]
