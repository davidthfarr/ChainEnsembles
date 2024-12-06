"""chain.py

Contains a simple python script/example of running an LLM chain with the
data in the ./data/ path.

"""

import os
import argparse
import data_utils
from transformers import (
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    QuantoConfig,
)

from chain_ensembles import HuggingFaceLink
from chain_ensembles import OpenAILink
from chain_ensembles import LLMChain


def get_cli_args():
    """
    A helper function to get the command line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="LLM Chain Ensembles")
    parser.add_argument("-o", type=str, help="Output directory.", required=True)
    parser.add_argument(
        "-d",
        type=str,
        help='Dataset name: "SemEval2016", "misinfo", "ibc"',
        required=True,
    )
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="The number of samples to draw for classification.",
    )
    return parser.parse_args()


def main():
    """
    The main function for a simple program that runs an LLM Chain!

    """
    # Read CLI args
    args = get_cli_args()

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
        data_df = data_df.sample(n=args.n)

    # Get all the prompts
    data_df["prompts"] = data_df.apply(
        lambda row: prompter.simple(**row), axis=1
    ).tolist()

    chain = [
        HuggingFaceLink(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            model_class=AutoModelForCausalLM,
            labels=labels,
            quantization_config=QuantoConfig('int4'),
        ),
        HuggingFaceLink(
            model_name="google/flan-ul2",
            model_class=T5ForConditionalGeneration,
            labels=labels,
            quantization_config=QuantoConfig('int4'),
        ),
        OpenAILink(model_name="gpt-4o", labels=labels),
    ]

    llm_chain = LLMChain(chain)
    llm_chain.run_chain(data_df, args.o, CoT=[False, False, False])
    data_utils.calculate_metrics(args.o, column_map)
    return 0


if __name__ == "__main__":
    main()
