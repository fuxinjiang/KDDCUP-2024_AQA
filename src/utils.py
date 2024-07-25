import dill
from tqdm import tqdm
from data_preprocessing import AQAdata
import faiss
import os.path as osp
import sys
import torch
import numpy as np
import json
import random
from sentence_transformers import SentenceTransformer

device = "cuda"

class model_embedding:
    
    def __init__(self, device="cuda", model_path=""):
        
        self.device = torch.device(device)
        self.embedding = SentenceTransformer(model_path, trust_remote_code=True).to(self.device)
    
    def __call__(self, sentence):
        
        embeddings = self.embedding.encode(sentence, normalize_embeddings=True, convert_to_tensor=True)
        return embeddings

class model_retriever:
    
    def __init__(self, data, device, path, model_path):
        
        self.dict = data.doc
        self.batch_size = 3
        self.index = faiss.read_index(path)
        self.ids = list(data.doc.keys())
        self.index = faiss.index_cpu_to_all_gpus(self.index)
        self.embedding = model_embedding(device, model_path)
    
    def encode(self, query):
        return self.embedding(query)
    
    def search(self, query, k):
        _, ids = self.index.search(query, k)
        result = []
        for line in ids:
            result.append([self.ids[id] for id in line])
        return result