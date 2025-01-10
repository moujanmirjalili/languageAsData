import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, context_length,allowed_special = True):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        if allowed_special:
            token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        else:
            token_ids = tokenizer.encode(txt)
        # Use a sliding window to chunk the book into overlapping sequences of context_length
        total = len(token_ids) - context_length
        for i in range(0, total):
            input_sequence = token_ids[i:i + context_length]
            
            #shift to the right
            target_sequence = token_ids[i + 1: i + context_length + 1]

            # input and output are represented as tensors
            self.input_ids.append(torch.tensor(input_sequence))
            self.target_ids.append(torch.tensor(target_sequence))
            if i %20000 == 0:
                print('\r', 'Create Dataset', i, '/',total, end='')

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(txt,allowed_special=True,tokenizer=tiktoken.get_encoding("gpt2"),
                      batch_size=8, context_length=4, shuffle=True, drop_last=True,
                         num_workers=0):



    # Create dataset
    dataset = GPTDataset(txt, tokenizer, context_length,allowed_special=allowed_special)
    train, dev, test = torch.utils.data.random_split(dataset, [0.8,0.1,0.1])
    
    # Create dataloader
    train_dataloader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    dev_dataloader = DataLoader(
        dev,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return train_dataloader, dev_dataloader, test_dataloader