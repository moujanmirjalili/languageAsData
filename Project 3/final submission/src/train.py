import torch
from ultOld import evaluate_accuracy
def train(model, train_loader, val_loader,optimizer, device, epochs=3, lr=3e-5,is_dry=False):
    optimizer = torch.optim.AdamW(model.score.parameters(), lr=lr) 
    global_step = 0 
    all_loss = []
    all_avg_loss = []
    all_acc_train = []
    all_acc_val = []
    
    

    for epoch in range(epochs): 
        model.train() 
        total_loss = 0.0 
        total_steps = len(train_loader)
        correct = 0 
        total = 0
        for batch_idx, batch in enumerate(train_loader): 
            input_ids = batch["input_ids"].to(device) 
            attention_mask = batch["attention_mask"].to(device) 
            labels = batch["labels"].to(device) 

            optimizer.zero_grad() 
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels) 
            loss = outputs.loss 
            loss.backward() 
            optimizer.step() 

            total_loss += loss.item() 
            all_loss.append(loss.item)
            global_step += 1 
            
            #Calculate Acc 
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx+1) % 10 == 0: 
                print(f"\rEpoch {epoch+1}/{epochs}, step {batch_idx+1}/{total_steps}, loss = {loss.item():.4f}",end="")
                if is_dry:
                    break
                #break
            if(batch_idx+1) % 500 == 0:
                print("")
                #break
                # Only print gradient information in the classification head (debug) 
                # for name, param in model.score.named_parameters(): 
                #     if param.requires_grad and param.grad is not None: 
                #         print(f"  [grad debug] {name}, grad mean: {param.grad.mean():.6f}") 
                #         break 

        avg_loss = total_loss / len(train_loader) 
        all_avg_loss.append(avg_loss)
        train_accuracy = correct / total if total > 0 else 0
        all_acc_train.append(train_accuracy)
        print(f"\nEpoch {epoch+1}/{epochs}, Average training loss: {avg_loss:.4f}") 
    
        # Validation accuracy 
        val_acc = evaluate_accuracy(model, val_loader, device) 
        all_acc_val.append(val_acc)
        print(f"Validation accuracy: {val_acc*100:.2f}% train_acc={train_accuracy*100:.2f}%\n") 

    return model, all_avg_loss,all_acc_train,all_acc_val
def train_head_only(model, train_loader, val_loader, device, epochs=3, lr=3e-5,is_dry=False): 
    optimizer = torch.optim.AdamW(model.score.parameters(), lr=lr)
    return train(model, train_loader, val_loader,optimizer, device, epochs=epochs, lr=3e-5,is_dry=is_dry)

def train_partial_unfreeze(model, train_loader, val_loader, device, epochs=3, lr=3e-5,is_dry=False): 
    params_to_optimize = [p for p in model.parameters() if p.requires_grad] 
    optimizer = torch.optim.AdamW(params_to_optimize, lr=lr) 
    return train(model, train_loader, val_loader,optimizer, device, epochs=epochs, lr=3e-5,is_dry=is_dry)
  