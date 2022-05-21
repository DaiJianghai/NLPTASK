from lib2to3.pgen2 import token
from multiprocessing import reduction
import os
from re import A
from numpy import mask_indices
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
from torchcrf import CRF
import transformers
from transformers import BertModel, BertTokenizer


def build_corpus(split, make_vocab=True, data_dir=r"D:\DevelopmentProgress\Project_VSCode\deeplearning\src\task\IE\Pipleline\NER\data"):
    """读取数据"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    with open(os.path.join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    word_lists = sorted(word_lists, key=lambda x: len(x), reverse=False)
    tag_lists = sorted(tag_lists, key=lambda x: len(x), reverse=False)

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        tag2id = build_map(tag_lists)
        tag2id['<PAD>'] = len(tag2id)
        return word_lists, tag_lists, tag2id
    else:
        return word_lists, tag_lists

def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps

class MyDataset(Dataset):
    def __init__(self, max_len, word_lists, tag_lists, tag2idx, tok):
        self.tokenizer = tok
        self.max_len = max_len
        self.word_lists = word_lists
        self.tag_lists = tag_lists
        self.tag2idx = tag2idx

    def __getitem__(self,index):
        # data = self.word_lists[index]
        # tag  = self.tag_lists[index]
        # # data_index = [self.word_2_index.get(i,self.word_2_index["<UNK>"]) for i in data]
        # # tag_index  = [self.tag_2_index[i] for i in tag]
        # dic = self.tokenizer.encode_plus(data, add_special_tokens=True, max_length=self.max_len, pad_to_max_length=True, truncation=True)
        # input_ids = dic["input_ids"]
        # token_type_ids = dic["token_type_ids"]
        # attention_mask = dic["attention_mask"]
        # tokens_len = len(data)
        # tag_index = [self.tag2idx.get(i,self.tag2idx["<PAD>"]) for i in tag] + [self.tag2idx["<PAD>"]] * (self.max_len - tokens_len)
        # return torch.tensor(input_ids,dtype=torch.int64),torch.tensor(attention_mask,dtype=torch.int64),torch.tensor(token_type_ids,dtype=torch.int64),torch.tensor(tag_index,dtype=torch.long)
        text = self.word_lists[index]
        label = self.tag_lists[index]
        ids = []
        token_type_ids = []
        attention_mask = []
        lbs = []
        # avoid word piece
        for i, word in enumerate(text):
            cur_label = self.tag2idx[label[i]]
            tokenized_word = self.tokenizer.encode(word, add_special_tokens=False)
            ids.extend(tokenized_word)
            lbs.extend([cur_label] * len(tokenized_word))
        # truncate
        ids = ids[:self.max_len-2]
        lbs = lbs[:self.max_len-2]
        # special token
        ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["cls_token"])] + ids + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["sep_token"])]
        lbs = [self.tag2idx["<PAD>"]] + lbs + [self.tag2idx["<PAD>"]]
        # padding
        attention_mask = [1] * len(ids) + [0] * (self.max_len - len(ids))
        token_type_ids = [0] * len(ids) + [0] * (self.max_len - len(ids))
        ids = ids + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map["pad_token"])] * (self.max_len - len(ids))
        lbs = lbs + [self.tag2idx["<PAD>"]] * (self.max_len - len(lbs))
        return torch.tensor(ids, dtype=torch.int64), torch.tensor(token_type_ids, dtype=torch.int64), torch.tensor(attention_mask, dtype=torch.int64), torch.tensor(lbs, dtype=torch.long)
            


    def __len__(self):
        return len(self.word_lists)



class Mymodel(nn.Module):
    def __init__(self, num_class, bert_path):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.crf = CRF(num_class, batch_first=True)
        self.linear = nn.Linear(768, num_class)
        self.drop = nn.Dropout(0.5)


    def forward(self, input_ids, token_type_ids, attention_mask, batch_tag=None):
        x = self.bert(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask)
        x = self.linear(x.last_hidden_state)
        x = self.drop(x)
        self.pre = torch.tensor(self.crf.decode(x)).reshape(-1)
        if batch_tag is not None:
            loss = -1 * self.crf(x, batch_tag, reduction='mean', mask=attention_mask.byte())
            return loss
        

def test(model, device, idx2tag, tokenizer): 
    model.eval()
    while True:
        text = input("请输入：")
        dic = tokenizer.encode_plus(add_special_tokens = False, pad_to_max_len = False)
        for k,v in dic:
            dic[k] = v.to(device)
        model.forward(dic["input_ids"],dic["token_type_ids"],dic["attention_mask"])
        pre = [idx2tag[i] for i in model.pre]

        print([f'{w}_{s}' for w,s in zip(text,pre)])




if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_data,train_tag,tag_2_index = build_corpus("train",make_vocab=True)
    dev_data,dev_tag = build_corpus("dev",make_vocab=False)
    index_2_tag = [i for i in tag_2_index]

    class_num  = len(tag_2_index)

    epoch = 5
    train_batch_size = 4
    dev_batch_size = 4
    hidden_num = 768
    bi = True
    lr = 5e-5
    bert_path = r"E:\Huggingface_Model\BERT\chinese-bert-wwm"
    max_len = 128
    num_tag = len(tag_2_index)

    tokenizer = BertTokenizer.from_pretrained(bert_path)
    train_dataset = MyDataset(max_len, train_data, train_tag, tag_2_index, tokenizer)
    train_dataloader = DataLoader(train_dataset,train_batch_size,shuffle=True, drop_last=True)

    dev_dataset = MyDataset( max_len, dev_data, dev_tag, tag_2_index, tokenizer)
    dev_dataloader = DataLoader(dev_dataset, dev_batch_size, shuffle=True)

    model = Mymodel(num_tag, bert_path)
    opt = torch.optim.Adam(model.parameters(),lr = lr)
    model = model.to(device)

    
    min_loss = 1e10
    for e in range(epoch):
        model.train()
        for input_ids, token_type_ids, attention_mask, batch_tag in tqdm(train_dataloader):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_tag = batch_tag.to(device)

            train_loss = model.forward(input_ids, token_type_ids, attention_mask, batch_tag)
            train_loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        all_pre = []
        all_tag = []
        cum_loss = 0
        for input_ids, token_type_ids, attention_mask, batch_tag in tqdm(dev_dataloader):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_tag = batch_tag.to(device)
            dev_loss = model.forward(input_ids, token_type_ids, attention_mask, batch_tag)
            all_pre.extend(model.pre.detach().cpu().numpy().tolist())
            all_tag.extend(batch_tag.detach().cpu().numpy().reshape(-1).tolist())
            cum_loss += dev_loss.item()
        score = f1_score(all_tag,all_pre,average="micro")
        print(f"{e},f1_score:{score:.3f},dev_loss:{dev_loss:.3f},train_loss:{train_loss:.3f}")
        if cum_loss < min_loss:
            min_loss = cum_loss
            torch.save(model.state_dict(),"model.pth")
            print(f"epoch:{e}  loss:{cum_loss}  score:{score} best model saved")
        if e == epoch - 1:
            print(classification_report(all_tag,all_pre))
    test(model, device, index_2_tag, tokenizer)
