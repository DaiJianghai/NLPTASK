import transformers
from transformers import BertTokenizer, BertForMaskedLM
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

class IMDBDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self.read(data_path)
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.data['input_ids'][idx],
            'attention_mask': self.data['attention_mask'][idx],
            'token_type_ids': self.data['token_type_ids'][idx],
        }

    def read(self, data_path):
        data = pd.read_csv(data_path)
        data["review"] = data["review"].apply(lambda x: x.lower().strip("\""))
        sentences = data["review"].values.tolist()
        sentences = self.tokenizer(sentences, add_special_tokens=True, max_length=self.max_len, padding ='max_length', truncation=True, return_tensors = 'pt')
        return sentences