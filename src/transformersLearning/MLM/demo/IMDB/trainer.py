import torch
import torch.nn as nn
from transformers import AdamW
from tqdm import tqdm

class Trainer:
    def __init__(self, model, dataloader, tokenizer, mlm_probability=0.15, lr=1e-4, with_cuda=True, cuda_devices=None, log_freq=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.is_parallel = False
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.log_freq = log_freq
        
        # 多GPU训练
        if with_cuda and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUS for BERT")
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
            self.is_parallel = True
        self.model.train()
        self.model.to(self.device)
        self.optim = AdamW(self.model.parameters(), lr=1e-4)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        
    def train(self, epoch):
        self.iteration(epoch, self.dataloader)
        
    def iteration(self, epoch, dataloader, train=True):
        str_code = 'Train'
        total_loss = 0.0
        for i,batch in tqdm(enumerate(dataloader), desc="Training"):
            b = batch["input_ids"].clone()
            inputs, labels = self._mask_tokens(b)
            batch['input_ids'] = inputs
            for key in batch.keys():
                batch[key] = batch[key].to(self.device)
            labels = labels.to(self.device)
            output = self.model(**batch, labels=labels)
            loss = output.loss.mean()
            
            if train:
                self.model.zero_grad()
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                
            total_loss += loss.item()
            post_fix = {
                "iter": i,
                "ave_loss": total_loss/(i+1)
            }
            if i % self.log_freq == 0:
                print(post_fix)
                
        print(f"EP{epoch}_{str_code},avg_loss={total_loss/len(dataloader)}")
        
    def _mask_tokens(self, inputs):
        """ Masked Language Model """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
            
        labels = inputs.clone()
        # 使用mlm_probability填充张量
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # 获取special token掩码
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        # 将special token位置的概率填充为0
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            # padding掩码
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            # 将padding位置的概率填充为0
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        # 对token进行mask采样
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # loss只计算masked
        
        # 80%的概率将masked token替换为[MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10%的概率将masked token替换为随机单词
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        # 余下的10%不做改变
        return inputs, labels
    

