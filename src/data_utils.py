""" data_utils.py

This module contains some basic helper functions for handling data.

"""

import os
import torch
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from prompters.misinfo_prompter import MisinfoPrompter
from prompters.stance_prompter import StancePrompter
from prompters.ibc_prompter import IBCPrompter


def get_cli_args():
    """
    A helper function to get the command line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="LLM Chain Ensembles")
    parser.add_argument('-o', type=str,
                        help='Output directory.', required=True)
    parser.add_argument('-d', type=str,
                         help='Dataset name: "SemEval2016", "misinfo", "ibc"', 
                         required = True)
    parser.add_argument('-n', type=int, default=10,
                       help='The number of samples to draw for classification.')
    return parser.parse_args()

def get_data_args(dataset_name):
    """
    A function that gets all the data arguments of the user provided dataset 
    name. We hard code the paths here. This is meant to be called from the 
    top level directory.

    Args:
        - dataset_name (str): The name of the dataset.

    Returns:
        Tuple of 
            - data_path (str): Path to local data.
            - column_map (Dict[str:str]): Dictionary that maps DF column names
                to common data schema name for each feild.
            - label_map (Dict[str:str]): Dictionary that maps the name of each
                label in a DF to an alternate naming convention. 
            - prompter (Prompter): A prompter from the ./prompters mini-library
                designed specifically for the dataset name provided.
    """
    if dataset_name == "SemEval2016":
        data_path = "./data/SemEval2016_all.csv"
        column_map = {"label": "Stance", "text": "Tweet", "target": "Target"}
        label_map = {"FAVOR": "for", "AGAINST": "against", "NONE": "neutral"}
        prompter = StancePrompter(label_map.values(), column_map)

    elif dataset_name == "misinfo":
        data_path = "./data/misinfo.tsv"
        column_map = {"label": "gold_label", "text": "headline"}
        label_map = {"misinfo": "fake", "real": "real"}
        prompter = MisinfoPrompter(label_map.values(), column_map)

    elif dataset_name == "ibc":
        data_path = "./data/ibc.csv"
        column_map = {"label": "leaning", "text": "sentence"}
        label_map = {"Liberal": "liberal", "Neutral": "neutral", "Conservative": "conservative"}
        prompter = IBCPrompter(label_map.values(), column_map)

    else:
        data_path, column_map, label_map, prompter = None, None, None, None

    return data_path, column_map, label_map, prompter

def read_data_to_df(path, column_map = None, label_map = None):
    """
    Helper function to read DataFrame from path.
    """
    if is_csv_path(path):
        try:
            raw_df = pd.read_csv(path)

        except UnicodeDecodeError:
            raw_df = pd.read_csv(path, encoding="ISO-8859-1")

    elif is_tsv_path(path):
        try:
            raw_df = pd.read_csv(path, delimiter = "\t")

        except UnicodeDecodeError:
            raw_df = pd.read_tsv(path, delimiter = "\t", encoding="ISO-8859-1")

    else:
        try:
            raw_df = pd.read_table(path)

        except UnicodeDecodeError:
            raw_df = pd.read_table(path, encoding="ISO-8859-1")

    # Rename labels if neccesary
    if column_map is not None and label_map is not None:
        lab_col = column_map["label"]
        raw_df[lab_col] = raw_df[lab_col].apply(lambda x: label_map[x])

    return raw_df

def is_csv_path(path):
    """
    Helper function to see if path has csv handle.
    """
    return True if path[-4:] == ".csv" else False

def is_tsv_path(path):
    """
    Helper function to see if path has tsv handle.
    """
    return True if path[-4:] == ".tsv" else False

def write_json(path, data):
    """Write dict to json."""
    with open(path, "w") as f:
        json.dump(data, f)
        
def read_json(path):
    """Reads json file to dict."""
    with open(path, "r") as f:
        data = json.load(f)
    
    return data

def concat_chain_results(df_list):
    """ Concatinates resulting DFs 

    Args:
        - df_list (List(pd.DataFrames)): List of dataframes with "retain"
            boolean.
    """
    for link_num, df in enumerate(df_list):
        df["link"] = link_num

    retains_df = pd.concat([df[df["retain"]] for df in df_list])
    return retains_df

def calculate_metrics(output_dir, column_map):
    """ A helper function to calculate metrics for all chain links after chain
        main in llm_chain.py.
    
    """
    meta_path = os.path.join(output_dir, "meta.json")
    meta_data = read_json(meta_path)

    df_names = [f"link_{lin}" for lin in range(meta_data["chain_len"])]
    df_names.append("final")

    for df_name in df_names:
        df_path = os.path.join(output_dir, df_name + "_df.pkl")
        df = pd.read_pickle(df_path)
        
        if df_name != "final":
            df = df[df["retain"]]
            
        meta_data[df_name + "_retain_accuracy"] = get_accuracy(df, column_map)
        meta_data[df_name + "_retain_f1_macro"] = get_f1_macro(df, column_map)
        meta_data[df_name + "_retain_f1_weight"] = get_f1_weighted(df, column_map)

    meta_data["ensemble_acc"] = get_accuracy(df, column_map, "backward_label")
    meta_data["ensemble_f1_macro"] = get_f1_macro(df, column_map, "backward_label")
    meta_data["ensemble_f1_weighted"] = get_f1_weighted(df, column_map, "backward_label")
    write_json(meta_path, meta_data)

def get_accuracy(results_df, column_map, df_label_col="pred_label"):
    """Helper to get accuracy"""
    return accuracy_score(results_df[column_map["label"]], results_df[df_label_col])

def get_f1_macro(results_df, column_map, df_label_col="pred_label"):
    """Helper to get macro F1 score"""
    return f1_score(results_df[column_map["label"]], results_df[df_label_col], 
                    average = "macro")

def get_f1_weighted(results_df, column_map, df_label_col="pred_label"):
    """Helper to get weighted F1 score"""
    return f1_score(results_df[column_map["label"]], results_df[df_label_col], 
                    average = "weighted")
