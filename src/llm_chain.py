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
from chain_sim import backward_pass

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
        df = data_df.copy()
        df_link_list = []
        
        for link_num, link in enumerate(self.chain_list):
            print(f"\n[i] Link {link_num}: {link.model_name} in progress...\n")

            if CoT[link_num]:
                prompts = df["CoT_prompts"].tolist()

            else:
                prompts = df["prompts"].tolist()
                
            link.load_model()            
            link_out = link.get_labels(prompts, 
                                       CoT = CoT[link_num],
                                       verbose = verbose,
                                       max_response_len = max_response_len)
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
        final_df = final_df.drop(["forward", "retain"], axis = 1)

        for link_num, df in enumerate(df_link_list):
            is_available = final_df["link"] >= link_num
            final_df[f"link_{link_num}_label"] = df["pred_label"].where(is_available, None)
            final_df[f"link_{link_num}_conf_score"] = df["conf_score"].where(is_available, None)
            
        final_df = backward_pass(final_df, self.chain_len)
        
        # Save results
        df_path = os.path.join(output_dir, "final_df.pkl")
        meta_data["final_df_path"] = "final_df.pkl"
        final_df.to_pickle(df_path)
        data_utils.write_json(meta_path, meta_data)

        return final_df

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

    # Get all the prompts
    data_df["prompts"] = data_df.apply(lambda row: prompter.simple(**row), axis=1).tolist()

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
