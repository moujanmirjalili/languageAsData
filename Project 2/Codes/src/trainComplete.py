from src.train import train,train_attention
import torch.nn as nn
import torch 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
from importlib.metadata import version
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from src.dataset import GPTDataset
from src.dataset import create_dataloader

class TrainComplete:
    def __init__(self,text_path = "content/spa_wikipedia_2021_30K-sentences.txt",path_to_save_folder= "model/train_data",tokenizer=tiktoken.get_encoding("gpt2"),allowed_special=True, is_attention_training = False):
        self.text_path = "content/spa_wikipedia_2021_30K-sentences.txt"
        self.path_to_save_folder= path_to_save_folder
        self.tokenizer = tokenizer
        self.allowed_special =  allowed_special
        self.is_attention_training = is_attention_training
    def train(self,model,
              vocab_size,device,raw_text,train_run_label,
                print_every=75,evaluate_every=3000,optimizer=None,criterion=None,
              batch_size = 32,
              embedding_dim = 128,
              context_length = 32,
              num_epochs =  4
             ):
        if optimizer == None:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        if criterion == None:
            criterion = nn.CrossEntropyLoss()
        
        text_path = self.text_path
        path_to_save_folder = self.path_to_save_folder
        
        #train_dataloader, dev_dataloader, test_dataloader = create_dataloader(
        #    raw_text, batch_size=batch_size, 
        #    context_length=context_length, shuffle=True
        #)
        train_dataloader, dev_dataloader, test_dataloader = create_dataloader(raw_text,tokenizer = self.tokenizer,
                                                                              allowed_special=self.allowed_special,
                                                                              batch_size=batch_size, 
                                                                              context_length=context_length,
                                                                              shuffle=True
                                                                             )
        data_loader = train_dataloader
        if self.is_attention_training:
            (all_losses,train_losses,perplexities,all_perplex) = train_attention(model,
                                                       num_epochs,
                                                       optimizer,criterion,data_loader,
                                                       path_to_save_folder,
                                                       train_run_label,
                                                       vocab_size,device,
                                                       evaluate_every,dev_dataloader,
                                                       print_every)
        else:   
            (all_losses,train_losses,perplexities,all_perplex) = train(model,
                                                                   num_epochs,
                                                                   optimizer,criterion,data_loader,
                                                                   path_to_save_folder,
                                                                   train_run_label,
                                                                   vocab_size,device,
                                                                   evaluate_every,dev_dataloader,
                                                                   print_every)