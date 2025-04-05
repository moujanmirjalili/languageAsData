import copy
import time
import torch 
import torch.nn as nn
from src.lora_helper import *
from src.lora_model import *
import math
def train_model_full_finetune(model, train_loader, val_loader, device, epochs=3, lr=5e-5):
    """
    Fully fine-tune all parameters of GPT-2, including the classification head.
    """
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    val_accs = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        #for input_ids, attention_mask, y_batch in train_loader:
        total_steps = len(train_loader)
        for i, instance in enumerate(train_loader):
            input_ids = instance['input_ids']
            attention_mask = instance['attention_mask']
            y_batch = instance['labels']
            #-
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = forward_for_classification(model, input_ids, attention_mask, device)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i+1) % 5 == 0: 
                print(f"\rEpoch {epoch+1}/{epochs}, step {i+1}/{total_steps}, loss = {loss.item():.4f}",end="")
            if(i+1) % 500 == 0:
                print("")

        avg_loss = total_loss / len(train_loader)
        val_acc = calc_accuracy(val_loader, model, device)
        train_losses.append(avg_loss)
        val_accs.append(val_acc)

        print(f"Epoch={epoch+1}, Loss={avg_loss:.4f}, ValAcc={val_acc*100:.2f}%")

    end_time = time.time()
    elapsed = end_time - start_time
    return elapsed, train_losses, val_accs


# %%----------------------------------------------------

def train_model_lora(
    model, 
    train_loader, 
    val_loader, 
    device, 
    epochs=3, 
    lr=1e-4, 
    log_grad_norms=False
):
    freeze_original_parameters(model)
    print_trainable_parameters(model)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler = amp.GradScaler()

    train_losses, val_accs = [], []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        #for step, (input_ids, attention_mask, y_batch) in enumerate(train_loader):
        total_steps = len(train_loader)
        for i, instance in enumerate(train_loader):
            input_ids = instance['input_ids']
            attention_mask = instance['attention_mask']
            y_batch = instance['labels']
            #-
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            with amp.autocast():
                logits = forward_for_classification(model, input_ids, attention_mask, device)
                loss = loss_fn(logits, y_batch)

            scaler.scale(loss).backward()

            # If the user wants to log gradient norms for educational demonstration:
            if log_grad_norms:
                scaler.step(optimizer)
                scaler.update()
                #show_gradient_norms(model)
            else:
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()
            if (i+1) % 5 == 0: 
                print(f"\rEpoch {epoch+1}/{epochs}, step {i+1}/{total_steps}, loss = {loss.item():.4f}",end="")
            if(i+1) % 500 == 0:
                print("")

        avg_loss = total_loss / len(train_loader)
        val_acc  = calc_accuracy(val_loader, model, device)
        train_losses.append(avg_loss)
        val_accs.append(val_acc)
        print(f"Epoch={epoch+1}, Loss={avg_loss:.4f}, ValAcc={val_acc*100:.2f}%")

    end_time = time.time()
    elapsed = end_time - start_time
    return elapsed, train_losses, val_accs