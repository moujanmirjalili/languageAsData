import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import pandas as pd
import os 


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=None):
        self.data = pd.read_csv(csv_file,sep=",")
        
        # Verify necessary columns in the CSV file
        required_columns = ["Text", "Label"]
        print(self.data.columns)
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"CSV file must contain the following columns: {required_columns}")
        
        # Ensure labels are of integer type
        self.data["Label"] = self.data["Label"].astype(int)
        
        self.texts = self.data["Text"].tolist()
        self.labels = self.data["Label"].tolist()
        
        # Set pad_token_id, if not specified, use tokenizer's pad_token_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
        
        # Encode texts
        self.encoded_texts = []
        for text in self.texts:
            try:
                encoded = tokenizer.encode(text, add_special_tokens=True)
                self.encoded_texts.append(encoded)
            except Exception as e:
                raise ValueError(f"Error encoding text: {text[:50]}...") from e
        
        # Dynamically calculate max_length, or use specified max_length
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]
        print("Max Length is: ",self.max_length)
        
        # Pad all sequences and generate attention_mask
        self.padded_texts = []
        self.attention_masks = []
        for enc in self.encoded_texts:
            enc = enc[:self.max_length]
            attention_mask = [1] * len(enc)
            
            pad_len = self.max_length - len(enc)
            if pad_len > 0:
                enc += [self.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
            
            self.padded_texts.append(enc)
            self.attention_masks.append(attention_mask)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.padded_texts[idx], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_masks[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        text = self.texts[idx]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
            "text": text
        }
    
    def _longest_encoded_length(self):
        return max(len(encoded_text) for encoded_text in self.encoded_texts)




def get_enc_dataset(data_path,tokenizer,batch_size=8,max_length=None,is_log = True):


    # Add a separate pad_token
    if tokenizer.pad_token is None:
        # Use '<|PAD|>' as the padding token
        tokenizer.add_special_tokens({'pad_token': '<|PAD|>'})
        pad_token_id = tokenizer.pad_token_id
        print("Added new pad_token '<|PAD|>' with ID:", pad_token_id)
    pad_token_id = tokenizer.pad_token_id

    # Create datasets
    if max_length:
        train_dataset = SpamDataset(os.path.join(data_path,"our_train.csv"),
                                    tokenizer,
                                    pad_token_id=pad_token_id,
                                    max_length= max_length)
    else:
        train_dataset = SpamDataset(os.path.join(data_path,"our_train.csv"),
                                    tokenizer,
                                    pad_token_id=pad_token_id)

    if not max_length:
        max_length=train_dataset.max_length
        
    val_dataset = SpamDataset(os.path.join(data_path,"our_dev.csv"), 
                              tokenizer, 
                              max_length=max_length,
                              pad_token_id=pad_token_id)
    test_dataset = SpamDataset(os.path.join(data_path,"our_test.csv"), tokenizer, max_length=train_dataset.max_length, pad_token_id=pad_token_id)
    print("Max Length: ", max_length)
    
    # Set DataLoader parameters
    num_workers = 0
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    
    print(f"Number of training batches: {len(train_loader)}, Number of validation batches: {len(val_loader)}")
    return train_dataset, val_dataset,test_dataset,train_loader,val_loader,test_loader, pad_token_id