"""llm_chain.py

Code that drives the methodology for the LLM Chain.

The developers realize that the pandas dependency is needless, and there are
better alternatives for production. However, this is an academic-focused package,
and it helps organize the results a bit.

We can revise to remove dependency in package becomes more widely used.
"""

import os
import json
import pandas as pd
from typing import List

from .hf_link import HuggingFaceLink
from .openai_link import OpenAILink
from .chain_sim import backward_pass


class LLMChain:
    """Chaining Together LLMs based on confidence scores.

    Attributes:
        chain_list (List[HuggingFaceLink] | OpenAILink]): A list of individual
            chain links from HuggingFaceLink or OpenAILink.
        chain_len (int): Then number of links in chain.

    """

    def __init__(self, chain_list: List[HuggingFaceLink | OpenAILink]):
        """Initializes an LLMChain

        Args:
            - chain_list (List[HuggingFaceLink | OpenAILink]): A list of individual
            chain links from HuggingFaceLink or OpenAILink.
        """
        self.chain_list = chain_list
        self.chain_len = len(chain_list)

    def run_chain(
        self,
        data_df: pd.DataFrame,
        output_dir: str,
        CoT: List[bool],
        verbose: bool = True,
        max_response_len: int = 500,
    ) -> pd.DataFrame:
        """Runs an LLM Chain for data labeling on pandas DataFrame.

        Args:
            data_df (pd.DataFrame): Dataframe of observations with columns:
                "prompts" with the LLM prompts.
            output_dir (str): The path to save output dataframe and meta data.
            CoT (List[bool]): List of boolean indicators to enable CoT for each
                chain link.
            verbose (bool): Boolean indicator to enable outputs.
            max_response_len (int): Used for CoT prompting.

        Returns:
            (pd.DataFrame): Dataframe with resulting chain data.
        """
        meta_data = {
            f"link_{i}": self.chain_list[i].model_name for i in range(self.chain_len)
        }
        meta_data["chain_len"] = self.chain_len
        meta_path = os.path.join(output_dir, "meta.json")
        write_json(meta_path, meta_data)
        df = data_df.copy()
        df_link_list = []

        for link_num, link in enumerate(self.chain_list):
            print(f"\n[i] Link {link_num}: {link.model_name} in progress...\n")

            if CoT[link_num]:
                prompts = df["CoT_prompts"].tolist()

            else:
                prompts = df["prompts"].tolist()

            link.load_model()
            link_out = link.get_labels(
                prompts,
                CoT=CoT[link_num],
                verbose=verbose,
                max_response_len=max_response_len,
            )
            link.unload_model()

            for key, value in link_out.items():
                df[key] = value

            n = len(df) // (self.chain_len - link_num)
            threshold = df["conf_score"].nlargest(n).min()
            meta_data[f"link_{link_num}_threshold"] = threshold

            df["retain"] = df["conf_score"] >= threshold
            df["forward"] = ~df["retain"]
            df["link"] = link_num

            link_df_path = os.path.join(output_dir, f"link_{link_num}_df.pkl")
            meta_data[f"link_{link_num}_path"] = f"link_{link_num}_df.pkl"
            df.to_pickle(link_df_path)
            print(f"\n[i] Link {link_num} complete. Saved to {link_df_path}.\n")

            df_link_list.append(df.copy())
            df = df[df["forward"]].copy()

        # Concat results and run backpass (rank based ensemble)
        final_df = pd.concat([df[df["retain"]] for df in df_link_list])
        final_df = final_df.drop(["forward", "retain"], axis=1)

        for link_num, df in enumerate(df_link_list):
            is_available = final_df["link"] >= link_num
            final_df[f"link_{link_num}_label"] = df["pred_label"].where(
                is_available, None
            )
            final_df[f"link_{link_num}_conf_score"] = df["conf_score"].where(
                is_available, None
            )

        final_df = backward_pass(final_df, self.chain_len)

        # Save results
        df_path = os.path.join(output_dir, "final_df.pkl")
        meta_data["final_df_path"] = "final_df.pkl"
        final_df.to_pickle(df_path)
        write_json(meta_path, meta_data)

        return final_df


#
# HELPER FOR WRITING TO JSON
#


def write_json(path, data):
    """Write dict to json."""
    with open(path, "w") as f:
        json.dump(data, f)
