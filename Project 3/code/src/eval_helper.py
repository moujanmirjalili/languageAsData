import re
import statistics
import json 
import os 
import torch
from time import gmtime, strftime
import numpy as np

from transformers import GPT2Tokenizer

path_to_save_folder = "model"
path_to_lora = os.path.join(path_to_save_folder,"lora")
path_to_partial = os.path.join(path_to_save_folder,"partial")
path_to_head_only = os.path.join(path_to_save_folder,"head")
path_to_full = os.path.join(path_to_save_folder,"full")
path_to_no_pre = os.path.join(path_to_save_folder,"no_pre")

# Example 
#    write_list_to_file("losses",train_losses,path_to_save_folder)





def write_list_to_file(label,list_to_save,path_to_save_folder):
    complete_path = os.path.join(path_to_save_folder,label+".json")
    with open(complete_path, 'w') as config_file:
        json.dump(list_to_save, config_file)

def read_list_from_file(label,path_to_save_folder):
    complete_path = os.path.join(path_to_save_folder,label+".json")
    with open(complete_path, 'r') as config_file:   
        loadedList = json.load(config_file)
    return loadedList


def save_everything(path_to_save_folder, train_run_label, elapsed, train_losses,train_acc, val_accs,train_acc_complete,val_acc_complete,test_acc_complete,model):
    path_to_save_folder = os.path.join(path_to_save_folder,train_run_label)
    if not os.path.exists(path_to_save_folder):
        os.makedirs(path_to_save_folder)
    write_list_to_file("losses",train_losses,path_to_save_folder)
    write_list_to_file("val_accs",val_accs,path_to_save_folder)
    write_list_to_file("elapsed",elapsed,path_to_save_folder)
    write_list_to_file("train_acc",train_acc,path_to_save_folder)
    #train_acc_complete,val_acc_complete,test_acc_complete
    write_list_to_file("train_acc_complete",train_acc_complete,path_to_save_folder)
    write_list_to_file("val_acc_complete",val_acc_complete,path_to_save_folder)
    write_list_to_file("test_acc_complete",test_acc_complete,path_to_save_folder)

    torch.save(model,  path_to_save_folder+"/"+"model_full")
    print("Everything saved at")
    print(strftime(" %H:%M:%S", gmtime()))#%Y-%m-%d



#Schould work now
def load_everything(path_to_save_folder,train_run_label,load_model = True):
    path_full = os.path.join(path_to_save_folder,train_run_label)
    
    losses = read_list_from_file("losses",path_full)
    val_accs= read_list_from_file("val_accs",path_full)
    elapsed = read_list_from_file("elapsed",path_full)

    train_acc = read_list_from_file("train_acc",path_full)
    train_acc_complete = read_list_from_file("train_acc_complete",path_full)
    val_acc_complete = read_list_from_file("val_acc_complete",path_full)
    test_acc_complete = read_list_from_file("test_acc_complete",path_full)

    if load_model:
        model = model = torch.load(path_full+"/"+"model_full", weights_only=False)
    else:
        model = None
    return (losses, val_accs, elapsed,train_acc,train_acc_complete,val_acc_complete,test_acc_complete,model)


def sliding_window(li,window,func):
    resultL = []
    for x in range(len(li)):
        start = x - round(window/2)
        if start < 0:
            start = 0
        stop = x + int(window/2)
        if stop >= len(li):
            stop = len(li) - 1 
        resultL.append(func(li[start:stop]))
    return resultL

def average_list(li,window):
    return sliding_window(li,window,np.mean)

def standart_deviation(li,window):
    temp = sliding_window(li,window,np.std)
    return np.mean(temp)



#------------------------
def classify_text(text, model, tokenizer, device, max_length, pad_token_id=50256): 
    model.eval() 
    enc = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length) 
    att_mask = [1]*len(enc) 
    pad_len = max_length - len(enc) 
    if pad_len > 0: 
        enc += [pad_token_id]*pad_len 
        att_mask += [0]*pad_len 
     
    input_ids = torch.tensor([enc], dtype=torch.long).to(device) 
    attention_mask = torch.tensor([att_mask], dtype=torch.long).to(device) 

    with torch.no_grad(): 
        #print(input_ids.shape)
        
        outputs = model(input_ids)#, attention_mask=attention_mask) 
        logits = outputs.logits 
        predicted = torch.argmax(logits, dim=-1).item() 
    return predicted


def print_classify(text_l, model,tokenizer,device):
    correct = 0
    total = 0
    for text, expected in text_l:
        c_result = classify_text(text,
                                 model,
                                 tokenizer,
                                 device,
                                 max_length=65,
                                 pad_token_id= tokenizer.pad_token_id )
        if c_result==1:
            s_result = "P"
        elif c_result == 0:
            s_result = "N"
        else: 
            raise ValueError("Output is not Valid")
        if expected=="1\n":
            ex_s = "P"
        elif expected =="0\n" : 
            ex_s = "N"
        else:
            raise ValueError("Exp not Valid")
        if s_result == ex_s:
            is_correct = "Yes"
            correct += 1 
        else:
            is_correct = "No "
        total += 1 
            
        print(s_result," Exp:",ex_s ,"Is correct:",is_correct,"=>", text)
    print("In Total ",correct,"/",total,"where correct")
def classify_all(path_to_save_folder, train_run_label, is_pad_token_eos):
    max_length =  65
    losses, val_accs, elapsed,train_acc,train_acc_complete,val_acc_complete,test_acc_complete, model = load_everything(path_to_save_folder,train_run_label,load_model =True)
    path_to_easy = "content/easy.csv"#
    path_to_hard = "content/hard.csv"
    easy_l = []
    with open(path_to_easy,"r") as f:
        for l in f.readlines():
            spli = l.split(";")
            t = (spli[0],spli[1])
            easy_l.append(t)

    hard_l = []
    with open(path_to_hard,"r") as f:
        for l in f.readlines():
            spli = l.split(";")
            t = (spli[0],spli[1])
            hard_l.append(t)
    
    model_name = "gpt2"
    DEBUG = False
    # Load GPT-2 Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if is_pad_token_eos:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        if tokenizer.pad_token is None:
            # Use '<|PAD|>' as the padding token
            tokenizer.add_special_tokens({'pad_token': '<|PAD|>'})
            #pad_token_id = tokenizer.pad_token_id
            #print("Added new pad_token '<|PAD|>' with ID:", pad_token_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEBUG:
        device = "cpu"

    print_classify(easy_l, model,tokenizer,device)
    print("\n\n")
    print_classify(hard_l, model,tokenizer,device)
            