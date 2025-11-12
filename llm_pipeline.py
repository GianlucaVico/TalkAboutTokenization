from transformers import pipeline
import torch
from vllm import LLM, SamplingParams

class _pipeline:
    def __new__(cls, model_name: str, quant: bool = False):
        if not hasattr(cls, 'model'):            
            if quant:
                cls.model = pipeline("text-generation", model="unsloth/Llama-3.3-70B-Instruct-bnb-4bit", device_map="auto")                
            else:
                cls.model = LLM(model_name, tensor_parallel_size=torch.cuda.device_count(), dtype='auto')
        return cls.model