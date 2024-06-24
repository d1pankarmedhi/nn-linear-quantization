<div align="center";>
<h1>qllm</h1>
<h3>🌩 Lightweight model quantization tool </h3>
</div>



The motivation behind this tool is to reduce model size and memory usage for inference. Storing large models can be burdensome, especially when dealing with edge devices. A model with a smaller footprint often simplifies and streamlines the inference process.

**qllm** tool allows developers to quantize any Hugging Face model from the HF Hub without losing much in terms of performance and accuracy. The underlying method is a simple linear quantization technique (both asymmetric and symmetric), enabling quick and ready-to-go quantization on the fly.


> **Note**: This tool is not ready for production environments and should only be used for building intuition. Consider it a tool for model quantization experimentation.

## Getting started

Download and load any model from huggingface. This example uses `facebook/opt-350m` model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch 

model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```
```
# The Original footprint of the model: 662.392832 MB
```

Run and test the original model.
```python
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe("What are we having for dinner?"))

# [{'generated_text': "What are we having for dinner?\nI'm having a steak and a salad.\nI'm"}]
```

Load and replace the Linear Layers of the model by qllm's LL layers and run quantization.

```python
from qllm import LinearQuantizer
from qllm.layers import W8A16LL

LinearQuantizer.replace_and_quantize_modules(model, W8A16LL, ['lm_head'])
```
```
# Model footprint after quantization: 359.799808 MB
```

Test the model after quantization
```python
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe("What are we having for dinner?"))

# [{'generated_text': "What are we having for dinner?\nI'm having a steak dinner.\nI'm having a"}]

```

## Observation

It is around `46%` reduction in model footprint which is outstanding. This may not be the same for every model but reduction of anywhere around `30%` is significantly helpful from the inference point of view.

