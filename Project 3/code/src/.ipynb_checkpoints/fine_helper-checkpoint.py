
import os 
from pathlib import Path 
import pandas as pd 

def load_dataset_from_path(data_path,file_name):
    return pd.read_csv(os.path.join(data_path,file_name), sep="\t", header=None, names=["Text", "Label"]) 


def create_balanced_dataset(df): ## spam = 0 ;; ham = 1 
    num_spam = df[df["Label"] == "0"].shape[0] 
    ham_subset = df[df["Label"] == "1"].sample(num_spam, random_state=123) 
    return pd.concat([ham_subset, df[df["Label"] == "0"]], ignore_index=True) 


#Just splits the dataset into two sets not like the original code
#
def random_split(df, train_frac): 
    df = df.sample(frac=1, random_state=123).reset_index(drop=True) 
    train_end = int(len(df) * train_frac) 
    #validation_end = train_end + int(len(df) * validation_frac) 
    train_df = df[:train_end] 
    validation_df = df[train_end:] 
    #test_df = df[validation_end:] 
    return train_df, validation_df

def load_complete_dataframe(data_path,is_balanced = True, is_log=True): 
    train_df = load_dataset_from_path(data_path,"train.tsv")
    validation_df = load_dataset_from_path(data_path,"dev.tsv")
    test_df = load_dataset_from_path(data_path,"test.tsv")
    print("Original dataset label counts:") 
    print(train_df["Label"].value_counts()) 
    print(train_df.shape)
    #--
    if is_balanced:
        balanced_train_df = create_balanced_dataset(train_df) 
        balanced_train_df["Label"] = balanced_train_df["Label"].map({"1": 1, "0": 0}) 
        print("\nBalanced dataset label distribution:") 
        print(balanced_train_df["Label"].value_counts()) 
        print(f"\nTraining set size: {len(train_df)}, Validation set size: {len(validation_df)}, Test set size: {len(test_df)}") 
    #-------------Save the file to csv
    train_df = balanced_train_df
    our_train_df, our_validation_df = random_split(balanced_train_df, 0.96) 
    #our_test_df = validation_df.sample(frac=1, random_state=123).reset_index(drop=True) 
    our_test_df = validation_df[1:]
    
    our_train_df.to_csv(os.path.join(data_path,"our_train.csv"), index=None) 
    our_validation_df.to_csv(os.path.join(data_path,"our_dev.csv"), index=None) 
    our_test_df.to_csv(os.path.join(data_path,"our_test.csv"), index=None) 

    return train_df, validation_df, test_df