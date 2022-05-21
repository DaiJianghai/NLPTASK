import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
 
class MyData(Dataset):
  def __init__(self, data):
      self.data = data
  def __len__(self):
      return len(self.data)
  def __getitem__(self, idx):
      length = len(self.data[idx])
      return self.data[idx]
 
def collate_fn(data):
    data.sort(key=lambda x: len(x), reverse=True)
    # seq_len = [s.size(0) for s in data] # 获取数据真实的长度
    # data = pad_sequence(data, batch_first=True)
    # data = pack_padded_sequence(data, seq_len, batch_first=True)   # pad_sequence + pack_padded_sequence = pack_sequence
    data = pack_sequence(data)
    return data
 
a = torch.tensor([1, 2, 3, 4]) # sentence 1
b = torch.tensor([5, 6, 7]) # sentence 2
c = torch.tensor([7, 8]) # sentence 3
d = torch.tensor([9]) # sentence 4
 
train_x = [a, b, c, d]
 
data = MyData(train_x)
data_loader = DataLoader(data, batch_size=2, shuffle=True, collate_fn=collate_fn)
 
batch_x = iter(data_loader)
for i in range (2):
 print(batch_x.next())