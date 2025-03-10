from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import torch

class EmbeddingModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.load_model(model_name)  # Load your model here
        self.embedding_dim = 768  # Dimension of embeddings (adjust if needed)

    def load_model(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
        model = AutoModel.from_pretrained(model_name)
        
        # Ensure that the tokenizer has a pad token, or set it to eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to eos_token
        return model, tokenizer

    def encode(self, model_name, texts):
        tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)

        # Ensure padding is handled correctly
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to eos_token if not defined
        
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        # Check if you are using the model correctly
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Example of pooling strategy
        return embeddings.numpy()
