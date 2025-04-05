# Plotting the Loss and Perplexity
import matplotlib.pyplot as plt
from src.eval_helper import average_list
import numpy


def plot_two(list1,label1,list2,label2,axLabel1 = None, axLabel2 = None,save = False):
    if not axLabel1:
        axLabel1 = ("Epoch","Loss")
    if not axLabel2:
        axLabel2 = ("Epoch","Perplexity")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(list1, label=label1) #linestyle='dashed',marker="o"
    plt.title(label1)
    plt.xlabel(axLabel1[0])
    plt.ylabel(axLabel1[1])
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(list2, label=label2,)
    plt.title(label2)
    plt.xlabel(axLabel2[0])
    plt.ylabel(axLabel2[1])
    plt.legend()
    if save:
        plt.savefig('temp.png')
    plt.show()


def get_prep_func(start_point, window_size = None):

    def func(li,ep):
        y_points = li[start_point:]
        x_points = numpy.linspace(0,ep,len(li))[start_point:]
        if window_size:
            y_points =  average_list(y_points,window_size)
        return (x_points,y_points)
            

    return func



def plot_multiple(first_graphs,label1,second_graphs,label2,axLabel1 = None, axLabel2 = None,save = False,first_y_lim = None,second_y_lim = None, main_label=None):

    if not axLabel1:
        axLabel1 = ("Epoch","Loss")
    if not axLabel2:
        axLabel2 = ("Epoch","Perplexity")
    plt.figure(figsize=(12, 5))
    if main_label:
        plt.suptitle(main_label, fontsize=16)
    plt.subplot(1, 2, 1)
    for label,x_labels,y_labels in first_graphs:
        plt.plot(x_labels,y_labels, label=label) #linestyle='dashed',marker="o"
    plt.title(label1)

    plt.xlabel(axLabel1[0])
    plt.ylabel(axLabel1[1])
    if first_y_lim:
        plt.ylim(top=first_y_lim)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for label,x_labels,y_labels in second_graphs:
        plt.plot(x_labels,y_labels, label=label) #linestyle='dashed',marker="o"
    plt.title(label2)
    plt.xlabel(axLabel2[0])
    plt.ylabel(axLabel2[1])
    if second_y_lim:
        plt.ylim(second_y_lim)
        #plt.ylim(top=second_y_lim)
        #plt.ylim(bottom=0)
    plt.legend()
    if save:
        plt.savefig('temp.svg')
    plt.show()
    