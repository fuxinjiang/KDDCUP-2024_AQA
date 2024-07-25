import os
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import tqdm
import json
import random
import math

class EmbeddingDataset(Dataset):
    def __init__(self,
            train_data_path,
            tokenizer,
            neg_nums,
            query_max_len = 128,
            passage_max_len = 512,
        ):

        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len        
        
        self.neg_nums = neg_nums
        self.train_data=self.read_train_data(train_data_path)
        if len(self.train_data[0].keys())==2:
            self.data_type='pair'
        elif len(self.train_data[0].keys())==3:
            self.data_type='triplet'
        self.tokenizer = tokenizer


    def read_train_data(self, train_data_path):
        train_data = []
        with open(train_data_path) as f:
            all_data = json.load(f)
        
        for data_dic in tqdm.tqdm(all_data):
            if 'pos' in data_dic and 'neg' not in data_dic:    
                return all_data                
    
            elif 'pos' in data_dic and 'neg' in data_dic:
                
                for text_pos in data_dic['pos']:
                    temp_dic = {}
                    temp_dic['query'] = data_dic['query']
                    temp_dic['pos'] = text_pos
                    
                    if len(data_dic['neg']) < self.neg_nums:
                        num = math.ceil( self.neg_nums / len(data_dic['neg']))
                        temp_dic['neg'] = random.sample(data_dic['neg'] * num, self.neg_nums )
                    else:
                        temp_dic['neg'] = random.sample(data_dic['neg'], self.neg_nums )

                    train_data.append(temp_dic)
        print(len(train_data))
        return train_data
    
    
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self,idx):
        return self.train_data[idx]

    def triplet_collate_fn(self,batch):

        all_querys = []
        all_pos_docs = []
        all_neg_docs = []

        for item in batch:
            all_querys.append(item['query'])
            all_pos_docs.append(item['pos'])
            all_neg_docs.extend(item['neg'])


        all_query_tokens = self.tokenizer(all_querys,padding='max_length',truncation=True,
                            max_length=self.query_max_len,return_tensors='pt')

        all_pos_doc_tokens = self.tokenizer(all_pos_docs,padding='max_length',truncation=True,
                            max_length=self.passage_max_len,return_tensors='pt')

        all_neg_doc_tokens = self.tokenizer(all_neg_docs,padding='max_length',truncation=True,
                            max_length=self.passage_max_len,return_tensors='pt')


        toekns_batch={}

        toekns_batch['query_input_ids'] = all_query_tokens['input_ids']
        toekns_batch['query_attention_mask'] = all_query_tokens['attention_mask']

        toekns_batch['pos_doc_input_ids'] = all_pos_doc_tokens['input_ids']
        toekns_batch['pos_doc_attention_mask'] = all_pos_doc_tokens['attention_mask']

        toekns_batch['neg_doc_input_ids'] = all_neg_doc_tokens['input_ids']
        toekns_batch['neg_doc_attention_mask'] = all_neg_doc_tokens['attention_mask']

        return toekns_batch
        

    def pair_collate_fn(self,batch):

        all_querys = []
        all_pos_docs = []

        for item in batch:
            all_querys.append(item['query'])
            all_pos_docs.append(item['pos'])


        all_query_tokens = self.tokenizer(all_querys,padding='max_length',truncation=True,
                            max_length=self.query_max_len,return_tensors='pt')

        all_pos_doc_tokens = self.tokenizer(all_pos_docs,padding='max_length',truncation=True,
                            max_length=self.passage_max_len,return_tensors='pt')

        toekns_batch={}

        toekns_batch['query_input_ids'] = all_query_tokens['input_ids']
        toekns_batch['query_attention_mask'] = all_query_tokens['attention_mask']

        toekns_batch['pos_doc_input_ids'] = all_pos_doc_tokens['input_ids']
        toekns_batch['pos_doc_attention_mask'] = all_pos_doc_tokens['attention_mask']


        return toekns_batch

if __name__ == "__main__":

    pass