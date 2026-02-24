from transformers import pipeline
import torch
from vllm import LLM
from sentence_transformers import SentenceTransformer

class _pipeline:
    def __new__(cls, model_name: str, hf: bool = True):
        if not hasattr(cls, 'model'):
            if hf:
                cls.model = pipeline("text-generation", model=model_name, device_map="auto", dtype=torch.bfloat16)
            else:
                cls.model = LLM(model_name, pipeline_parallel_size=torch.cuda.device_count(), dtype=torch.bfloat16, trust_remote_code=True, enable_prefix_caching=True)
        return cls.model

class _sentence_encoder:
    def __new__(cls, model_name: str = 'all-mpnet-base-v2'):
        if not hasattr(cls, 'encoder'):            
            cls.encoder = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
        return cls.encoder