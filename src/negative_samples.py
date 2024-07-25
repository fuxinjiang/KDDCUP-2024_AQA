# %%
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
from utils import model_retriever, model_embedding

path_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
# %%
# 首先构建papers的索引
def build_papers_database(data, model_path, device, output_path):
    
    doc = data.doc
    emb_model = model_embedding(device, model_path)
    batch_size = 2  # Adjust the batch size according to your memory capacity
    doc_values = list(doc.values())
    embedding_list = []
    for i in tqdm(range(0, len(doc_values), batch_size)):
        batch = doc_values[i:i + batch_size]
        batch_embeddings = emb_model(batch).cpu()
        embedding_list.append(batch_embeddings)
        del batch_embeddings
        
    # Concatenate all the embeddings
    doc_embedding_vectors = torch.cat(embedding_list, dim=0)    
    d = 1024
    # 修改为L2距离
    index = faiss.IndexFlatL2(d)
    index.add(doc_embedding_vectors.numpy())
    faiss.write_index(index, output_path)
    
def build_negative_sample(data, model_path, device, vectordb_path, output_path):
    # adjust the faiss index path and model path here
    print("加载retrieval")
    retriever = model_retriever(data, 
                              device=device, 
                              path=vectordb_path, 
                              model_path=model_path)
    print("加载retrieval完毕")
    emb_batch_size = 2
    searching_batch_size = 100
    embeddings = []
    result = []
    
    queries = data.question["train"]
    passage_ids = list(data.doc.keys())
    
    print("开始进行embedding")
    length = len(queries)
    for k in tqdm(range(length // emb_batch_size + 1)):
        start = k*emb_batch_size
        end = k*emb_batch_size+emb_batch_size
        if end>length:
            end=length
        if start >= end:
            break
        sentences = [item["query"] for item in queries[start:end]]
        embs = retriever.encode(sentences)
        embeddings.append(embs.detach().cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    
    print("embedding结束")
    print("开始检索")
    for k in tqdm(range(length // searching_batch_size + 1)):
        start = k * searching_batch_size
        end = k * searching_batch_size+searching_batch_size
        if end > length:
            end=length
        if start >= end:
            break
        query_embeddings = embeddings[start:end, :]
        
        # embeddings.append(self.embedding(sentence))
        # embs = self.embedding(sentence, batch_size=end-start)
        res = retriever.search(query_embeddings, k=150)
        result += res
    
    print("检索结束")
    #with open(osp.join(output_path, 'rerank_dataset.json'),'w') as f: 
     
    output_data = []  
    for res, query in zip(result, queries):
        pos_pid = query["pids"]
        query_question = query["query"]
        pos_set = set(pos_pid)
        random_select = random.sample(passage_ids, k=100)
        neg_0 = random.sample(list(set(res[30:90])-pos_set), k=20)
        neg_1 = random.sample(list(set(res[90:150])-pos_set), k=20)
        neg_2 = random.sample(list(set(random_select)-pos_set-set(res)), k=20)
        
        waiting_lists = [neg_0, neg_1, neg_2]
        pos_samples = []
        neg_samples = []
        
        for wait_list in waiting_lists:
            
            for pid in wait_list:
                neg_samples.append(data.doc[pid])
                
        for pos_pid in pos_set:
            pos_samples.append(data.doc[pos_pid])
        
        output_data.append({"query": query_question, "pos": pos_samples, "neg": neg_samples})

    with open(osp.join(output_path, 'train_dataset.json'), 'w') as f:
        json.dump(output_data, f, ensure_ascii=False)
            
# %%
if __name__ == '__main__':
    
    # %%
    print("开始构建paper向量数据库")
    with open(osp.join(path_root, 'data/process_data', f'AQAdata.dill'), 'rb') as f:
        data:AQAdata = dill.load(f)
    device ="cuda"
    model_path = "../model/Alibaba-NLP/gte-large-en-v1.5"
    output_path = "../data/process_data/database.index"
    build_papers_database(data, model_path, device, output_path)
    print("构建paper向量数据库完毕")
    # %%
    print("开始构建正负样本数据集")
    output_fold_path = "../data/process_data"
    build_negative_sample(data, model_path, device, output_path, output_fold_path)
    print("构建正负样本数据集完毕")