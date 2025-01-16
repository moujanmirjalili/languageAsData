import torch

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def forward_for_classification(model, input_ids, attention_mask, device):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    return logits



@torch.no_grad()
def calc_accuracy(loader, model, device, max_batches=None):
    model.eval()
    correct, total = 0, 0
    #for i, instance in enumerate(loader):
        #print(instance)
    #for i, (input_ids, attention_mask, y_batch,text) in enumerate(loader):
    for i, instance in enumerate(loader):
        input_ids = instance['input_ids']
        attention_mask = instance['attention_mask']
        y_batch = instance['labels']
        #print(input_ids.shape)
        #print(torch.max(input_ids))
        #print(attention_mask.shape)
        text = instance['text']
        if max_batches and (i+1) > max_batches:
            break

        input_ids = input_ids.to(device)
        #input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        y_batch = y_batch.to(device)

        logits = forward_for_classification(model, input_ids, attention_mask, device)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    return correct / total if total > 0 else 0





def advanced_metrics(loader, model, device):
    """
    Calculate precision, recall, F1-score, and accuracy.
    """
    model.eval()
    preds_list, labels_list = [], []
    for input_ids, attention_mask, y_batch in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        y_batch = y_batch.to(device)

        with torch.no_grad():
            logits = forward_for_classification(model, input_ids, attention_mask, device)
        preds = torch.argmax(logits, dim=-1)
        preds_list.extend(preds.cpu().numpy())
        labels_list.extend(y_batch.cpu().numpy())

    accuracy  = np.mean(np.array(preds_list) == np.array(labels_list))
    precision = precision_score(labels_list, preds_list)
    recall    = recall_score(labels_list, preds_list)
    f1        = f1_score(labels_list, preds_list)
    return accuracy, precision, recall, f1