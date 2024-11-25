"""chain_simulations.py

A file that provides functionality for chaining together existing fully labeled
datasets.

"""

import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score


def get_permutations(
    df_list: List[pd.DataFrame],
    df_names: List[str],
    chain_len: int,
    truth_col_name: str,
    n_trails: int = 1,
    verbose: bool = True,
    backward: bool = False,
) -> pd.DataFrame:
    """Performs and LLM labeling experiment on on all permutations of data

    We assume all datasets used in the simulation come from the labeling function.
    It's critical that the dataset columns are in the same order from the labeler.
    If you are using the llm_label.py script, then this is handled for you automatically.

    Args:
        df_list (List[pd.DataFrame]): A list of dataframes which at a minimum
            contain the 'conf_score' and 'pred_label' columns.
        df_names (List[str]): A list of the model dataframe names in the same
            order provided as df_list. Example: ['llama3', 'flan', 'gpt4o']
        chain_len (int): How long to make the chains for each experiment.
        truth_col_name (str): The name of the column which contains the
            ground truth labels for the dataset.
        n_trails (int): The number of times to randomly select splits for making
            combinations.
        backward (bool): Boolean indicator for backward pass.
        verbose (bool): Boolean indicator to enable tqdm output.

    Returns:
        (pd.Dataframe): Results of the chaining experiment across chain_len permutations of
            all models df_list.
    """
    permutations = list(itertools.permutations(zip(df_list, df_names), chain_len))

    results_dict = defaultdict(list)

    for perm in tqdm(permutations, disable=(not verbose)):
        link_list = [p[0] for p in perm]
        link_order = [p[1] for p in perm]
        results_dict["link_order"].append("-->".join(link_order))

        for i in range(chain_len):
            results_dict[f"link_{i}"].append(link_order[i])

        random_f1, random_acc = [], []
        for _ in range(n_trails):
            random_df, random_meta = chain_dataframes(
                link_list, truth_col_name, random_sample=True
            )
            random_f1.append(random_meta["total_f1"])
            random_acc.append(random_meta["total_acc"])

        results_dict["random_acc"].append(np.mean(random_acc))
        results_dict["random_f1"].append(np.mean(random_f1))

        final_df, meta_data = chain_dataframes(link_list, truth_col_name)
        results_dict["chain_acc"].append(meta_data["total_acc"])
        results_dict["chain_f1"].append(meta_data["total_f1"])

        if backward:
            bp_df = backward_pass(final_df, chain_len)
            bp_acc = accuracy_score(bp_df[truth_col_name], bp_df["backward_label"])
            bp_f1 = f1_score(
                bp_df[truth_col_name], bp_df["backward_label"], average="macro"
            )
            results_dict["back_rank_acc"].append(bp_acc)
            results_dict["back_rank_f1"].append(bp_f1)

    return pd.DataFrame(dict(results_dict))


def chain_dataframes(
    df_list: List[pd.DataFrame], truth_col_name: str, random_sample: bool = False
) -> Tuple[pd.DataFrame, dict]:
    """Chains dataframes and calculates key metrics for each link.

    Args:
        df_list (List[pd.DataFrame]): A list of dataframes which at a minimum
            contain the 'conf_score' and 'pred_label' columns. In the order
            in which you desire them to be chained. The dataframes should
            be simular such that all other columns (except 'pred_label' and
            'conf_score' have the same data in the same order.
        truth_col_name (str): The name of the column which contains the
            ground truth.
        random_sample (bool): Boolean indicator to random sample instead of
            select on conf_score.

    Returns:
        final_df (pd.DataFrame): Dataframe with results from chaining.
        meta_data (dict): metadata with accuracies and thresholds.
    """
    if not _check_truth_col_consistency(df_list, truth_col_name):
        print(
            "[!] The truth columns are misaligned. Please ensure the",
            "each dataframe in df_list contains the same data in the",
            "same oder!",
        )
        return None

    meta = {}
    df_link_list = []

    for link_num, df in enumerate(df_list):
        df = df.copy()
        prefix = f"link_{link_num}_full"
        meta = _get_acc_and_f1(df, truth_col_name, meta, prefix)
        df["link"] = link_num

        if df_link_list:
            for link_df in df_link_list:
                df = df[link_df["forward"]]

        n = len(df) // (len(df_list) - link_num)
        threshold = df["conf_score"].nlargest(n).min()
        meta[f"link_{link_num}_threshold"] = threshold

        if random_sample:
            retain_arr = np.array([True] * n + [False] * (len(df) - n))
            np.random.shuffle(retain_arr)
            df["retain"] = retain_arr

        else:
            df["retain"] = df["conf_score"] >= threshold

        df["forward"] = ~df["retain"]

        prefix = f"link_{link_num}_retained"
        meta = _get_acc_and_f1(df[df["retain"]], truth_col_name, meta, prefix)
        meta[prefix + "_size"] = len(df[df["retain"]])

        df_link_list.append(df)

    final_df = pd.concat([df[df["retain"]] for df in df_link_list])
    final_df = final_df.drop(["forward", "retain"], axis=1)

    for link_num, df in enumerate(df_list):
        is_available = final_df["link"] >= link_num
        final_df[f"link_{link_num}_label"] = df["pred_label"].where(is_available, None)
        final_df[f"link_{link_num}_conf_score"] = df["conf_score"].where(
            is_available, None
        )

    meta = _get_acc_and_f1(final_df, truth_col_name, meta, "total")

    return final_df, meta


def backward_pass(
    final_df: pd.DataFrame, chain_len: int, rank: bool = True
) -> pd.DataFrame:
    """Normalizing and Maximizing Based on Link Available Scores.

    Args:
        - final_df (pd.DataFrame): A dataframe output from chain_dataframes()
        - chain_len (int): The length of the chain.
        - rank (bool): Use the relative rank to increase the

    Returns:
        (pd.DataFrame): A dataframe with the backward pass results.

    """
    df = final_df.copy()

    for i in range(chain_len):
        c_name = f"link_{i}_conf_score"

        if rank:
            raw_rank = df[c_name].rank(method="max")
            num_non_na = sum(pd.notna(df[c_name]))
            df[f"link_{i}_conf_rank"] = raw_rank / num_non_na

        else:
            valid_values = df[c_name].replace([np.inf, -np.inf], np.nan).dropna()
            min_val = min(valid_values)
            max_val = max(valid_values)
            normalized_val = (df[c_name] - min_val) / (max_val - min_val)
            df[f"link_{i}_conf_rank"] = normalized_val

    link_rank_arr = [df[f"link_{i}_conf_rank"] for i in range(chain_len)]
    link_label_src = np.nanargmax(link_rank_arr, axis=0)
    df["backward_label_link_src"] = link_label_src

    all_link_labels = np.array([df[f"link_{i}_label"] for i in range(chain_len)]).T
    backward_label = [
        all_link_labels[row][idx] for row, idx in enumerate(link_label_src)
    ]
    df["backward_label"] = backward_label

    return df


def _check_truth_col_consistency(df_list, truth_col_name):
    """
    Check if the truth_col_name column is the same for each df in df list.
    """
    first_df = df_list[0][truth_col_name]

    for df in df_list[1:]:
        if not (first_df == df[truth_col_name]).all():
            return False

    return True


def _get_acc_and_f1(df, truth_col_name, meta_data, col_prefix):
    """Retrives accuracy and f1 from df and writes to meta_data."""
    df = df.copy()
    acc = accuracy_score(df[truth_col_name], df["pred_label"])
    f1 = f1_score(df[truth_col_name], df["pred_label"], average="macro")
    meta_data[col_prefix + "_acc"] = acc
    meta_data[col_prefix + "_f1"] = f1
    return meta_data
