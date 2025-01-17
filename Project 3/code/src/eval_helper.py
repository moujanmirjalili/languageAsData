import re
import statistics
import json 
import os 

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


def save_everythin(path_to_save_folder, train_run_label, elapsed, train_losses, val_accs):
    path_to_save_folder = os.path.join(path_to_save_folder,train_run_label)
    if not os.path.exists(path_to_save_folder):
        os.makedirs(path_to_save_folder)
    write_list_to_file("losses",train_losses,path_to_save_folder)
    write_list_to_file("val_accs",val_accs,path_to_save_folder)
    write_list_to_file("elapsed",elapsed,path_to_save_folder)

#Schould work now
def load_everything(path_to_save_folder,train_run_label):
    path_full = os.path.join(path_to_save_folder,train_run_label)
    losses = read_list_from_file("losses",path_full)
    val_accs= read_list_from_file("val_accs",path_full)
    elapsed = read_list_from_file("elapsed",path_full)
    
    model = model = torch.load(path_full+"/"+"model_full", weights_only=False)
    return (losses, val_accs, elapsed)

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