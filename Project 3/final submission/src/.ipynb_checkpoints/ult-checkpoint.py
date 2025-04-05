# ult.py 

import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd
import torch
from transformers import GPT2LMHeadModel

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """
    Download and unzip the spam SMS dataset.
    
    Parameters:
        url (str): The download URL of the dataset.
        zip_path (str): The path to save the downloaded ZIP file.
        extracted_path (str): The path to the folder where the ZIP file will be extracted.
        data_file_path (Path): The full path to the data file after extraction.
    """
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    print(f"Downloading dataset from {url}...")
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    print(f"Dataset downloaded to {zip_path}.")

    print(f"Extracting {zip_path} to {extracted_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    print(f"Dataset extracted to {extracted_path}.")

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    if not original_file_path.exists():
        raise FileNotFoundError(f"Expected data file not found after extraction: {original_file_path}")

    os.rename(original_file_path, data_file_path)
    print(f"File renamed and saved as {data_file_path}")

def create_balanced_dataset(df):
    """
    Create a balanced dataset where the number of "ham" and "spam" samples are equal.
    
    Parameters:
        df (pd.DataFrame): The original dataset containing the "Label" column.
    
    Returns:
        pd.DataFrame: The balanced dataset.
    """
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]], ignore_index=True)
    return balanced_df

def random_split(df, train_frac, validation_frac):
    """
    Randomly split the dataset into training, validation, and test sets.
    
    Parameters:
        df (pd.DataFrame): The dataset to split.
        train_frac (float): The fraction of the dataset to use for training.
        validation_frac (float): The fraction of the dataset to use for validation.
    
    Returns:
        tuple: (train_df, validation_df, test_df)
    """
    df_shuffled = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df_shuffled) * train_frac)
    validation_end = train_end + int(len(df_shuffled) * validation_frac)
    train_df = df_shuffled[:train_end]
    validation_df = df_shuffled[train_end:validation_end]
    test_df = df_shuffled[validation_end:]
    return train_df, validation_df, test_df

def load_pretrained_gpt2(model_name="gpt2", device=torch.device("cpu")):
    """
    Load a pre-trained GPT-2 model and set it to evaluation mode.
    
    Parameters:
        model_name (str): The name of the pre-trained model.
        device (torch.device): The device to load the model onto (CPU or GPU).
    
    Returns:
        GPT2LMHeadModel: The loaded model.
    """
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model

def initialize_classifier_head(model):
    """
    Initialize the weights of the classification head (score layer).
    
    Parameters:
        model (transformers.GPT2ForSequenceClassification): The GPT-2 classification model.
    """
    if hasattr(model, 'score'):
        torch.nn.init.xavier_uniform_(model.score.weight)
        print("Classification head initialized.")
    else:
        raise AttributeError("Model does not have a 'score' layer.")

def classify_text(text, model, tokenizer, device, max_length, pad_token_id=50256):
    """
    Classify a single piece of text using the classification model.
    
    Parameters:
        text (str): The text to classify.
        model (transformers.GPT2ForSequenceClassification): The classification model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        device (torch.device): The device to perform computation on (CPU or GPU).
        max_length (int): The maximum sequence length.
        pad_token_id (int): The ID of the padding token.
    
    Returns:
        str: The classification result, either "spam" or "ham".
    """
    model.eval()
    enc = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
    att_mask = [1] * len(enc)
    pad_len = max_length - len(enc)
    if pad_len > 0:
        enc += [pad_token_id] * pad_len
        att_mask += [0] * pad_len

    input_ids = torch.tensor([enc], dtype=torch.long).to(device)
    attention_mask = torch.tensor([att_mask], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted == 1 else "ham"

def evaluate_accuracy(model, data_loader, device):
    """
    Evaluate the accuracy of the model on a given data loader.
    
    Parameters:
        model (transformers.GPT2ForSequenceClassification): The classification model.
        data_loader (torch.utils.data.DataLoader): The data loader.
        device (torch.device): The device to perform computation on (CPU or GPU).
    
    Returns:
        float: The accuracy of the model.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total if total > 0 else 0
    return accuracy

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
