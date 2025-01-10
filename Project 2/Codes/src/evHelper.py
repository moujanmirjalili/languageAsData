from src.helper import read_list_from_file
import os
import torch
import statistics
import numpy as np 
import numpy
def load_everything(path_to_save_folder,train_run_label):
    path_full = os.path.join(path_to_save_folder,train_run_label)
    losses = read_list_from_file("losses",path_full)
    step_losses= read_list_from_file("step_losses",path_full)
    perplexities = read_list_from_file("perplexities",path_full)
    all_perplex = read_list_from_file("all_perplex",path_full)
    model = model = torch.load(path_full+"/"+"model_full", weights_only=False)
    return (losses, step_losses, perplexities,all_perplex,model)

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


#def movingaverage(interval, window_size):
#    window = numpy.ones(int(window_size))/float(window_size)
#    return numpy.convolve(interval, window, 'same')