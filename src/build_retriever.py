# %%
import json
import torch
import time
import numpy as np
import os.path as osp
from glob import glob
from tqdm import tqdm
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
# %%
class VectorstoreBuilder:
    
    def __init__(self, db_type:str, model_path:str, data_path:str):
        
        self.type = db_type 
        self.model_path = model_path
        self.data_path = data_path

    def load_documents(self):
        
        doc_path = osp.join(self.data_path, 'process_data', 'documents.pt')
        if osp.exists(doc_path):
            doc_list = torch.load(osp.join(self.data_path, 'process_data', 'documents.pt'))
        else:
            raw_doc1 = json.load(open(osp.join(self.data_path, 'AQA', 'pid_to_title_abs_update_filter.json'), 'r'))
            raw_doc = {**raw_doc1}
            doc_list = []
            for doc_id in raw_doc.keys():
                if raw_doc[doc_id]['title'] is None:
                    title = ' '
                else:
                    title = raw_doc[doc_id]['title']
                doc = Document(page_content= str(title) + '\n\n' + str(raw_doc[doc_id]['abstract']),
                                          metadata={'pid':doc_id})
                doc_list.append(doc)
            torch.save(doc_list, doc_path)
        return doc_list

    def load_data(self):
        
        doc_list = self.load_documents()
        sentences, metadatas = [], []
        for doc in doc_list:
            sentences.append(doc.page_content)
            metadatas.append(doc.metadata)

        return sentences, metadatas

    def build_vectorstore_mutilGPU(self):
        
        sentences, metadatas = self.load_data()
        model = SentenceTransformer(self.model_path, trust_remote_code=True)
        model.max_seq_length = 1024
        pool = model.start_multi_process_pool()
        emb = model.encode_multi_process(sentences, pool, batch_size=384, normalize_embeddings=True)
        model.stop_multi_process_pool(pool)
        torch.save([sentences, emb, metadatas], osp.join(self.data_path, f'{self.type}_emb.bin'))
    
    
# %%
if __name__ == '__main__':
    
    # %%
    model_path =  '../output/model'
    data_path = '../data/'
    builder = VectorstoreBuilder('gte', model_path, data_path)
    emb = builder.build_vectorstore_mutilGPU()
    # %%
