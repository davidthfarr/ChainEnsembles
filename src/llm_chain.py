""" llm_chain.py

Code that drives the methodology for the LLM Chain.

"""

import os
import json
import pandas as pd
from transformers import T5ForConditionalGeneration, AutoModelForCausalLM, BitsAndBytesConfig

import data_utils
from hf_link import HuggingFaceLink
from openai_link import OpenAILink

class LLMChain:
    """ Chaining Together LLMs based on confidence score!

    """
    def __init__(self, chain_list):
        """ Initializes an LLMChain

        Args:
            - chain_list (List[HuggingFaceLink | OpenAILink]): A list of class
                link instantiations with methods .load_model.
        """
        self.chain_list = chain_list
        self.chain_len = len(chain_list)
        
    def run_chain(self, data_df, output_dir, CoT, verbose = True, 
                  max_response_len = 500):
        """ Runs a full LLM Chain for data labeling on a data frame.

        Args:
            - data_df (pd.DataFrame): Dataframe of observations with columns:
                "prompts" with the LLM prompts.
            - output_dir (str): The path to save the output to.
            - CoT (List[bool]): List of boolean indicators to enable CoT for each
                chain link. CURRENTLY NOT SUPPORTED.
            - verbose (bool): Boolean indicator to enable outputs.
            - max_response_len (int): Used for CoT prompting. CURRENTLY NOT SUPPORTED.

        Returns: None.
        """
        meta_data = {f"link_{i}": self.chain_list[i].model_name for i in range(self.chain_len)}
        meta_data["chain_len"] = self.chain_len
        meta_path = os.path.join(output_dir, "meta.json")
        data_utils.write_json(meta_path, meta_data)
        
        for link_num, link in enumerate(self.chain_list):
            print(f"\n[i] Link {link_num}: {link.model_name} in progress...\n")

            if CoT[link_num]:
                prompts = data_df["CoT_prompts"].tolist()

            else:
                prompts = data_df["prompts"].tolist()
                
            link.load_model()            
            link_out = link.get_labels(prompts, 
                                       CoT = CoT[link_num],
                                       verbose = verbose,
                                       max_response_len = max_response_len)
            link.unload_model()

            for key, value in link_out.items():
                data_df[key] = value

            n = len(data_df) // (self.chain_len - link_num)
            threshold = data_df["conf_score"].nlargest(n).min()
            
            data_df["retain"] = data_df["conf_score"] >= threshold
            data_df["forward"] = ~data_df["retain"]

            link_df_path = os.path.join(output_dir, f"link_{link_num}_df.pkl")
            meta_data[f"link_{link_num}_path"] = f"link_{link_num}_df.pkl"
            data_df.to_pickle(link_df_path)
            print(f"\n[i] Link {link_num} complete. Saved to {link_df_path}.\n")
            
            data_df = data_df[data_df["forward"]].copy()

        df_list = []
        for link_num in range(self.chain_len):
            df_path = os.path.join(output_dir, f"link_{link_num}_df.pkl")
            df_list.append(pd.read_pickle(df_path))

        all_retains = data_utils.concat_chain_results(df_list)
        df_path = os.path.join(output_dir, "all_retained_df.pkl")
        meta_data["all_retained_path"] = "all_retained_df.pkl"
        all_retains.to_pickle(df_path)
        data_utils.write_json(meta_path, meta_data)

        return all_retains

def main():
    """
    The main function for a simple program that runs an LLM Chain!
    
    """
    # Read CLI args
    args = data_utils.get_cli_args()

    # Get HF and OpenAI tokens
    hf_token = os.getenv("HF_TOKEN")
    openai_token = os.getenv("OPENAI_API_KEY")

    # Ensure outpath exists
    if not os.path.exists(args.o):
        os.makedirs(args.o)

    data_path, column_map, label_map, prompter = data_utils.get_data_args(args.d)

    if data_path is None:
        print("[!] Dataset name invalid! Exiting Program.")
        return 1
        
    labels = list(label_map.values())
    data_df = data_utils.read_data_to_df(data_path, column_map, label_map)

    if args.n > 0 and args.n <= data_df.shape[0]:
        data_df = data_df.sample(n = args.n)

    # Get all the prompts and/or CoT prompts
    data_df["prompts"] = data_df.apply(lambda row: prompter.simple(**row), axis=1).tolist()
    data_df["CoT_prompts"] = data_df.apply(lambda row: prompter.CoT(**row), axis=1).tolist()

    chain = [HuggingFaceLink(model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
                             model_class = AutoModelForCausalLM, 
                             labels = labels,
                             quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                            ),
             HuggingFaceLink(model_name = "google/flan-ul2", 
                            model_class = T5ForConditionalGeneration, 
                            labels = labels, 
                            quantization_config = BitsAndBytesConfig(load_in_8bit=True),
                            ),
             OpenAILink(model_name="gpt-4o", labels = labels)
            ]
    
    llm_chain = LLMChain(chain)
    llm_chain.run_chain(data_df, args.o, CoT = [False, False, False])
    data_utils.calculate_metrics(args.o, column_map)
    return 0
    
if __name__ == "__main__":
    main()