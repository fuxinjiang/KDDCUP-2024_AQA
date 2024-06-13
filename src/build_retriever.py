import json
import torch
import time
import numpy as np
import logging
import sys
import argparse
import os.path as osp
from glob import glob
from tqdm import tqdm
from langchain_core.documents import Document
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.vectorstores.utils import DistanceStrategy
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from torch.nn import DataParallel
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

MODEL_PATH = '../model/nvidia/NV-Embed-v1'

class VectorstoreBuilder:
    def __init__(self, db_type:str, mnt_root:str):
        self.type = db_type 
        self.work_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
        self.mnt_root = mnt_root
    
    def load_documents(self):
        doc_path = osp.join(self.work_root, 'processed_data', 'documents.pt')
        if osp.exists(doc_path):
            doc_list = torch.load(osp.join(self.work_root, 'processed_data', 'documents.pt'))
        else:
            raw_doc1 = json.load(open(osp.join(self.work_root, 'AQA', 'pid_to_title_abs_new.json'), 'r'))
            raw_doc2 = json.load(open(osp.join(self.work_root, 'AQA', 'pid_to_title_abs_update_filter.json'), 'r'))
            raw_doc = {**raw_doc1, **raw_doc2}
            # raw_doc = raw_doc1

            doc_list = []
            for doc_id in raw_doc.keys():
                if raw_doc[doc_id]['title'] is None:
                    title = ' '
                else:
                    title = raw_doc[doc_id]['title']
                doc = Document(page_content= '# ' + title + '\n\n' + raw_doc[doc_id]['abstract'],
                                          metadata={'pid':doc_id})
                doc_list.append(doc)
            torch.save(doc_list, doc_path)

        return doc_list

    def build_vectorstore_bge(self, embedding_model_path):
        doc_list = self.load_documents()
        
        model_kwargs = {'device': 'cuda', 'trust_remote_code':True}
        encode_kwargs = {'normalize_embeddings': True, 'batch_size':16, 'show_progress_bar': True} # set True to compute cosine similarity
        embedding_hf = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            max_seq_length=1024,
        )

        vectorstore = FAISS.from_documents(doc_list, embedding_hf, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        vectorstore.save_local(osp.join(self.mnt_root, 'vector_db', f'{self.type}'))
        return vectorstore

    def build_vectorstore_mutilGPU(self, embedding_model_path):
        doc_list = self.load_documents()
        
        sentences, metadatas = [], []
        for doc in doc_list:
            sentences.append(doc.page_content)
            metadatas.append(doc.metadata)

        sentences = sentences[:2048]
        metadatas = metadatas[:2048]
        # model = SentenceTransformer(embedding_model_path, trust_remote_code=True)
        # model = AutoModel.from_pretrained(embedding_model_path, trust_remote_code=True)
        # model = AutoModel.from_pretrained('/mnt/bn/infralab-bytebm-energon/mlx/users/yong.fu/hf_models/models--nvidia--NV-Embed-v1/snapshots/8259ef97d6e2ea4b70141d025b9903a217041f3d/', trust_remote_code=True).to("cuda:0")
    
        # max_length = 2048
        # batch_size = 64

        start = time.time()
        # emb = model._do_encode(sentences, batch_size=batch_size, instruction="", max_length=max_length)
        emb = self.emb_batch_encode(sentences, "cuda:0")
        print(f"type emb: {type(emb)}, len: {len(emb)}")
        print(f"time: {time.time() - start}")

        torch.save([sentences, emb, metadatas], osp.join(self.mnt_root, f'{self.type}_emb.bin'))

    def emb_batch_encode_split(self, sentences, gpu_id):
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(gpu_id)
        max_length = 2048
        batch_size = 32

        chunk_size = 128
        chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]

        embs = []
        for i, chunk in enumerate(chunks):
            start = time.time()
            emb = model._do_encode(chunk, batch_size=batch_size, instruction="", max_length=max_length)
            print(f"gpu:{gpu_id} for chunk {i} time: {time.time() - start}")
            embs.append(emb)
            torch.cuda.empty_cache()

        final_emp = np.concatenate(embs, axis=0)
        return final_emp

    def emb_batch_encode(self, sentences, gpu_id):
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(gpu_id)
        max_length = 2048
        batch_size = 32

        logging.info(f"gpu: {gpu_id}")
        start = time.time()
        emb = model._do_encode(sentences, batch_size=batch_size, instruction="", max_length=max_length)
        logging.info(f"gpu:{gpu_id} encode time: {time.time() -start}")
        return emb

    def load_data(self):
        doc_list = self.load_documents()
        
        sentences, metadatas = [], []
        for doc in doc_list:
            sentences.append(doc.page_content)
            metadatas.append(doc.metadata)

        return sentences, metadatas


def emb_batch_encode(sentences, gpu_id):
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(f"cuda:{gpu_id}")
    max_length = 2048
    batch_size = 32

    start = time.time()
    emb = model._do_encode(sentences, batch_size=batch_size, instruction="", max_length=max_length)
    print(f"gpu:{gpu_id} encode time: {time.time() -start}")
    return emb

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    model_name =  "nvidia/NV-Embed-v1"
    builder = VectorstoreBuilder(model_name, "../output")
    sentences, metadatas = builder.load_data()
    num_gpus = torch.cuda.device_count()
    print(f"gpus: {num_gpus}")
    sentence_len = len(sentences)
    print(f"sentences len: {sentence_len}")
    print(f"gpu: {args.gpu}")
    chunk_size = sentence_len // num_gpus
    print(f"chunk_size: {chunk_size}")
    data_chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]

    data_gpu = data_chunks[args.gpu]
    emb = emb_batch_encode(data_gpu, args.gpu)
    mnt_root = '../output/'
    torch.save([sentences, emb, metadatas], osp.join(mnt_root, f'{model_name}_gpu_{args.gpu}_emb.bin'))
