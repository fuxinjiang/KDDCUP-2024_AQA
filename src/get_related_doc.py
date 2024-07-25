import torch
import faiss
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from typing import List
from sentence_transformers import SentenceTransformer

class BaseRetriever:
   
    def __init__(self, db_path:str, embedding_model_path:str, faiss_index_path:str):
        
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.sentences, emb, self.metadatas = torch.load(db_path)
        print('build faiss index')
        self.vectorstore = faiss.IndexFlatIP(1024) 
        self.vectorstore = faiss.IndexIDMap(self.vectorstore)
        self.vectorstore.add_with_ids(emb, np.arange(len(self.sentences)))
        print('add faiss successfully')
        faiss.write_index(self.vectorstore, faiss_index_path)
        self.query_embedder = SentenceTransformer(embedding_model_path, device=self.device, trust_remote_code=True)
        # self.sentences = np.array(self.sentences) # TODO: OOM bug
        self.metadatas = np.array(self.metadatas)
        print('baseRetriever init successfully')

    def get_related_doc(self, query:List[str], top_k:int):
    
        query = self.query_embedder.encode(query, normalize_embeddings=True, device=self.device)
        score, ids = self.vectorstore.search(query, k=top_k)
        return self.metadatas[ids]

    def get_pid(self, metadata_batch):
        
        metadata_batch = metadata_batch.tolist()
        doc_id_list = []
        for metadata in metadata_batch:
            temp_list = []
            for meta in metadata:
                temp_list.append(meta['pid'])
            doc_id_list.append(temp_list)
        return doc_id_list

    def __call__(self, query:str, top_k:int=20):
        # TODO: batch infer query
        metadata_batch = self.get_related_doc(query, top_k)
        doc_id_list = self.get_pid(metadata_batch)
        return doc_id_list


if __name__ == '__main__':
    
    with open(osp.join('../data/AQA', 'qa_test_wo_ans_new.txt'), 'r') as f:
        val_query_list = f.readlines()

    embedding_model_path = '../output/model'
    
    vectorstore_path = '../data/gte_emb.bin'
    
    faiss_index_path = osp.join('../data', f'train')

    retriever = BaseRetriever(vectorstore_path, embedding_model_path, faiss_index_path)
    
    file_write_obj = open(osp.join('../result', f"gte_result.txt"), 'a')

    for query in tqdm(val_query_list):
        query = eval(query)
        
        question = query['question']
        soup = BeautifulSoup(query.get('body', ''), 'html.parser')
        body = soup.get_text()
        retriever_input = "query:" + question + "\n\n" + body

        doc_id_list = retriever([retriever_input])
        file_write_obj.write(','.join(doc_id_list[0]))
        file_write_obj.write('\n')
    file_write_obj.close()