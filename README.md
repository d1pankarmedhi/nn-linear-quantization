# qllm
🌩 Light weight model quantization tool

The motivation of this tool is to reduce model size and memory usage for inference. Storing large models is often a hassle, especially when we are dealing with edge devices. Model with smaller footprint is often makes the job easy and effecient when it comes to inference processes.

**qllm** enables developers to quantize any huggingface model from the HF_hub without loosing much on the performance and accuracy. The underlining method is a simple linear quantization technique (both asymmetric and symmetric) enabling quick and ready to go quantiztion on the fly. 

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

