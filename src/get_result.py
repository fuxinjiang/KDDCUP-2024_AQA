import torch
import faiss
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from typing import List
from sentence_transformers import SentenceTransformer
from  transformers import AutoTokenizer, AutoModel
MODEL_PATH = '../model/nvidia/NV-Embed-v1'

class Retriever:
    
    def __init__(self, db_path:str, embedding_model_path:str, faiss_index_path:str):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        # hacking different length
        sentences, emb, metadatas = torch.load(db_path)
        self.sentences = sentences[:-3]
        self.metadatas = metadatas[:-3]
        print(f"emb length: {len(emb)}")
        print(f"sentences: {len(self.sentences)}")

        # self.sentences, emb, self.metadatas = torch.load(db_path)
        
        if not osp.exists(faiss_index_path):
            print('build faiss index')
            
            self.vectorstore = faiss.IndexFlatIP(4096) 
            self.vectorstore = faiss.IndexIDMap(self.vectorstore)

            print(f"emb length: {len(emb)}")
            print(f"sentences: {len(self.sentences)}")
            self.vectorstore.add_with_ids(emb, np.arange(len(self.sentences)))
            print('add faiss successfully')
            faiss.write_index(self.vectorstore, faiss_index_path)
        else:
            self.vectorstore = faiss.read_index(faiss_index_path)
        
        # if self.device == 'cuda':
        #     self.vectorstore = faiss.index_cpu_to_all_gpus(self.vectorstore)

        self.query_embedder = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to('cuda:0')

        # self.query_embedder = SentenceTransformer(embedding_model_path, device=self.device, trust_remote_code=True)
        # self.sentences = np.array(self.sentences) # TODO: OOM bug
        self.metadatas = np.array(self.metadatas)
        print('baseRetriever init successfully')

    def get_related_doc(self, query:List[str], top_k:int):
        max_length = 2048
        batch_size = 32

        query_emb = self.query_embedder._do_encode(query, batch_size=batch_size, instruction="", max_length=max_length)
        # query = self.query_embedder.encode(query, normalize_embeddings=True, device=self.device)
        score, ids = self.vectorstore.search(query_emb, k=top_k)
        return ids

    def get_pid(self, all_ids):
        doc_id_list = []
        for ids in all_ids:
            metadata_batch=self.metadatas[ids]
            id_list = [m["pid"] for m in metadata_batch]
            doc_id_list.append(id_list)

        return doc_id_list

    def __call__(self, query:List[str], top_k:int=20):
        # TODO: batch infer query
        ids = self.get_related_doc(query, top_k)
        all_doc_id_list = self.get_pid(ids)
        return all_doc_id_list


if __name__ == '__main__':
    
    work_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    mnt_root = '../output'
    with open(osp.join(work_root, 'AQA', 'qa_test_wo_ans_new.txt'), 'r') as f:
        val_query_list = f.readlines()
    ckpt_id = 1
    vectorstore_path = osp.join(mnt_root, 'nvidia/NV-Embed-v1_all_emb.bin')
    faiss_index_path = osp.join(mnt_root, 'nvidia', f'faiss_index_{ckpt_id}')
    retriever = Retriever(vectorstore_path, MODEL_PATH, faiss_index_path)

    all_retriever_input = []
    for query in tqdm(val_query_list):
        query = eval(query)
        
        question = query['question']
        soup = BeautifulSoup(query.get('body', ''), 'html.parser')
        body = soup.get_text()
        retriever_input = "# question:" + question + "\n\nDescription:" + body
        all_retriever_input.append(retriever_input)
    
    all_doc_id_list = retriever(all_retriever_input)
    with open(osp.join(work_root, 'result', f"nvidia_{ckpt_id}.txt"), 'w') as result_file:
        for id_list in all_doc_id_list:
            result_file.write(','.join(id_list))
            result_file.write('\n')
