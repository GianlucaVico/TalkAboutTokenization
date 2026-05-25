import torch
from transformers import pipeline

class _pipeline:
    def __new__(cls, model_name: str, *args):
        if not hasattr(cls, 'model'):                
            cls.model = pipeline("text-generation", model=model_name, device_map="auto", dtype=torch.bfloat16)
           
        return cls.model

class _sentence_encoder:
    def __new__(cls, model_name: str = 'all-mpnet-base-v2'):
        if not hasattr(cls, 'encoder'):           
            from sentence_transformers import SentenceTransformer 
            cls.encoder = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
        return cls.encoder
