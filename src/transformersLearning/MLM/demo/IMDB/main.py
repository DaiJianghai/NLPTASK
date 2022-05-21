from dataset import IMDBDataset
from trainer import Trainer
from config import Config
from transformers import BertTokenizer, BertForMaskedLM
import torch
from torch.utils.data import DataLoader
from torchsummary import summary


if __name__ == "__main__":
    config = Config()
    model = BertForMaskedLM.from_pretrained(config.model_path)
    tokenizer = BertTokenizer.from_pretrained(config.model_path)
    dataset = IMDBDataset(config.small, tokenizer, config.max_len)
    data_loader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    trainer = Trainer(model, data_loader, tokenizer)
    for epoch in range(config.epochs):
        trainer.train(epoch)