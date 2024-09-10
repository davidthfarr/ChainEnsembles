""" llm_label.py

Simple labeling approach to label an entire dataset with a single model.

"""


import os
import pandas as pd
import argparse
from transformers import BitsAndBytesConfig

import data_utils
from hf_link import HuggingFaceLink, MODELS_TESTED
from openai_link import OpenAILink

def get_cli_args():
    """
    A helper function to get the command line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-o', type=str, required=True,
                        help='Output directory.'
                       )
    parser.add_argument('-d', type=str, required = True,
                        help='Dataset name. Supported datasets are ' \
                             '"SemEval2016", "misinfo"', 
                       )
    parser.add_argument('-m', required=True, type=str,
                         help='The model to use for labeling. Supported models ' \
                          'are "gpt-4o", "gpt-4o-mini", "llama3-8B-instruct" ' \
                          '"phi3-medium", "mistral-7B-instruct".'
                        )
    parser.add_argument('-n', type=int, default=10,
                       help='The number of samples to draw for classification.' \
                            ' (default = 10)'
                       )
    parser.add_argument('--CoT', action=argparse.BooleanOptionalAction,
                        help='Flag to enable CoT prompting.')
    parser.add_argument('-q', type=int, required=False, default=8,
                        help='Model quantization to apply. Supported are "8" ' \
                             'and "4" for 8 and 4 bit respectivley. To load ' \
                             'with full precison specify -q != 8 or 4. ' \
                             '(default = 8)'
                        )
    return parser.parse_args()

MODELS = {
    "flan-ul2": "google/flan-ul2",
    "llama3-8B-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "phi3-medium": "microsoft/Phi-3-medium-128k-instruct",
    "mistral-7B-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini"
}

def main():
    """
    The entry point for the program.
    """
    # Read CLI args
    args = get_cli_args()

    # Ensure valid model
    if args.m not in MODELS.keys():
        print("[!] Invalid model selection! Use -h to veiw supported models.")
        return 1

    # Get HF and OpenAI tokens
    hf_token = os.getenv("HF_TOKEN")

    # Load data
    data_path, column_map, label_map, prompter = data_utils.get_data_args(args.d)
    
    if data_path is None:
        print("[!] Dataset name invalid! Exiting Program.")
        return 1
        
    labels = list(label_map.values())
    data_df = data_utils.read_data_to_df(data_path, column_map, label_map)

    # Sub-sample data
    if args.n > 0 and args.n <= data_df.shape[0]:
        data_df = data_df.sample(n = args.n)

    # Make prompts
    if args.CoT:
        data_df["prompts"] = data_df.apply(lambda row: prompter.CoT(**row), axis=1).tolist()

    else:
        data_df["prompts"] = data_df.apply(lambda row: prompter.simple(**row), axis=1).tolist()

    if args.m in ["gpt-4o", "gpt-4o-mini"]:
        model = OpenAILink(model_name=args.m, labels = labels)

    else:
        if args.q == 8:
            qunat_config = BitsAndBytesConfig(load_in_8bit=True)

        elif args.q == 4: 
            qunat_config = BitsAndBytesConfig(load_in_4bit=True)

        else:
            qunat_config = None

        model_name = MODELS[args.m]
        model = HuggingFaceLink(model_name = model_name,
                                model_class = MODELS_TESTED[model_name],
                                labels = labels,
                                hf_token = hf_token,
                                quantization_config = qunat_config
                               )

    model.load_model()
    model_out = model.get_labels(data_df["prompts"].tolist(), 
                                 CoT = args.CoT)
    model.unload_model()

    for key, value in model_out.items():
        data_df[key] = value

    # Ensure outpath exists
    if not os.path.exists(args.o):
        os.makedirs(args.o)
        
    df_path = os.path.join(args.o, "results_df.pkl")
    data_df.to_pickle(df_path)

    meta_data = vars(args)
    meta_data["results_path"] = "results_df.pkl"
    meta_data["accuracy"] = data_utils.get_accuracy(data_df, column_map)
    meta_data["f1_macro"] = data_utils.get_f1_macro(data_df, column_map)
    meta_path = os.path.join(args.o, "meta.json")
    data_utils.write_json(meta_path, meta_data)
    print(f"\n[i] Labeling complete. Resulting dataframe saved to {df_path} " \
          f"meta data saved to {meta_path}.")

    return 0

if __name__ == "__main__":
    main()
