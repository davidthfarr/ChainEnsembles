# LLM Chain Ensembles

This repository contains a python package and the code for the paper [LLM Chain Ensembles for Scalable and Accurate Data Annotation](https://arxiv.org/abs/2410.13006). LLM Chain Ensemble's use a sequence of LLMs to label subsets of data selected using uncertainty estimates. This method reduces zero-shot prediction costs by exposing limited data to high-cost models at the end of the chain and can yield increased performance. 

```mermaid
---
title: LLM Zero-Shot Prediction with LLM Chain Ensemble
---

flowchart LR
    A(Data) --> B{LLM 1}
    B --Lowest 1/3 Confidence--> C(Fowarded Data)
    B --High Confidence Examples--> G(Labeled Subset 1)
    C --> D{LLM 2}
    D --> H(Labeled Subset 2)
    D --Lowest 1/2 Confidence--> E(Fowarded Data)
    E --> F{LLM 3}
    F --> O(Labeled Subset 3)
    G --> M(Rank Based Ensemble)
    H --> M
    O --> M
    M --> Z(Final Label)
```

## Getting Started

### Installation

Install the most recent release with pip. 

```
uv pip install chain-ensembles
```

We recommend using [uv](https://github.com/astral-sh/uv) to manage your python environment and packages. It is much faster and cleaner.

```
uv venv
uv pip install chain-ensembles
```

### Authentication

The packages classes `HuggingFaceLink` and `OpenAILink` require some API tokens for authentication. These are accessed via environmental variables. If you are running from the terminal, please set the following environmental variables.

```bash
$ export HF_TOKEN=your_hf_token_here!
$ export OPENAI_API_TOKEN=your_openai_token_here!
```

### Example Scripts

There are some example scripts available in the [scripts/](./scripts/) directory of this repository. We used these to drive the experiments for the [paper](https://arxiv.org/abs/2410.13006). These scripts can be ran directly or serve as reference for any developer hoping to make their own chain ensembles! To run the scripts be sure to clone the repository and install the chain_ensembles package. Read more [here](#scripts).

## LLM Chain Ensembles

In this section, we will cover the basic functionality provided by the package.

### Using LLM Links To Label Data

The smallest data labeling class is a link! We provide two sources for data labeling links:
- Opensource models available on [Huggingface](https://huggingface.co/) with the [transformers](https://huggingface.co/docs/transformers) package. 
- Closed source models available through the [Open AI API](https://platform.openai.com/docs/overview).

The method needed to label data with any link is the `.get_labels()` method!

#### HuggingFaceLink Example

```python
labels = ["against", "for", "neutral"]
llama_link = HuggingFaceLink(
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    model_class = AutoModelForCausalLM,
    labels = labels,
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
)

prompts = ["Classify the stance toward something. I'm against something"]
data_out = llama_link.get_labels(prompts)
```

#### OpenAILink

```python
gpt4_link = OpenAILink(model_name="gpt-4o", labels = labels)

prompts = ["Classify the stance toward something. I'm against something"]
data_out = gpt4_link.get_labels(prompts)
```

### Chaining LLMs

Putting it all together now!

```python
import pandas as pd
from transformers import T5ForConditionalGeneration, AutoModelForCausalLM, BitsAndBytesConfig

from chain_ensembles import HuggingFaceLink, OpenAILink, LLMChain

data_df = pd.DataFrame({
    "prompts": ["Classify the stance toward something. I'm against something"]*12, 
    "Stance": ["against"]*12
})

labels = ["for", "against", "neutral"]

llama_link = HuggingFaceLink(
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
    model_class = AutoModelForCausalLM, 
    labels = labels,
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
)
flan_link = HuggingFaceLink(
    model_name = "google/flan-ul2", 
    model_class = T5ForConditionalGeneration, 
    labels = labels, 
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
)
gpt4_link = OpenAILink(model_name="gpt-4o", labels = labels)

llm_chain = LLMChain(chain_list=[llama_link, flan_link, gpt4_link])
CoT_setting = [False, False, False]
data_out = llm_chain.run_chain(data_df, "./chain_out", CoT_setting)
```

## Scripts

To run the scripts be sure to clone the repository and install the chain_ensembles package.

### Chain Example Script

To run an example chain ensemble we provide an python script in [chain_example.py](./src/chain_example.py) that runs a chain of LLama3-8B-instruct, Flan-UL2, and GPT-4o. The code has three command line arguments: 
- `-d` The dataset to use. The user is expected to enter "SemEval2016", "misinfo" or "ibc". The [SemEval2016](https://www.saifmohammad.com/WebPages/StanceDataset.htm) dataset is used for the stance detection task, the [misinfo](https://github.com/skgabriel/mrf-modeling) dataset is available for the misinformation detection task, and the [ibc](https://github.com/SALT-NLP/LLMs_for_CSS/tree/main/css_data/ibc) dataset is available for ideology detection task.
- `-o` The directory to output the chain results.
- `-n` The number of samples to select from said dataset. Specify -1 to use the entire dataset. Default is 10.

```bash
$ python scripts/chain_example.py -d SemEval2016 -o ./chain_out -n 20
```

### Labeling Entire Datasets and Simulating Chain Ensembles

We also provide an interface for labeling entire datasets with LLMs and analyzing their results post-hoc to run simulations of chain ensembles. To label entire datasets we provide a python script in [llm_label.py](./scripts/llm_label.py) that runs with the following command line arguments.

- `-m` The model to use to label the dataset. Select from  "flan-ul2", "llama3-8B-instruct", "phi3-medium", "mistral-7B-instruct", "gpt-4o", and "gpt-4o-mini".
- `-d` The dataset to label. The user is expected to enter "SemEval2016", "misinfo" or "ibc."
- `-o` The output directory to save labeled data.
- `-n` The number of samples to select from said dataset. Specify -1 to use the entire dataset. Default is 10.
- `q` Specify model quantization. Supported are "8" and "4" for 8 and 4 bit respectively. To load with full precision specify -q != 8 or 4. Default is  8.

```bash
python ./scripts/llm_label.py -m llama3-8B-instruct -d "SemEval2016" -o "./llama_test/" -n -1 -q 0
```

Once you have fully labeled datasets, you can run a simple simulation across them in a separate notebook or python script using the functions provided by the [chain_sim.py](./src/chain_ensembles/chain_sim.py) module. Set up the script with the following code.

```python
import pandas as pd
from chain_ensembles import get_combinations, chain_dataframes, backward_pass

llama_link = pd.read_pickle("./llama_test/results_df.pkl")
flan_link =  pd.read_pickle("./flan_test/results_df.pkl")
gpt_link =  pd.read_pickle("./gpt_test/results_df.pkl")

links = [llama_link, flan_link, gpt_link]
names = ["llama", "flan", "gpt", "mistral", "phi"]
```

To run a sinle chain ensemble iteration, use the `chain_dataframes` and `backward_pass` functions.

```python
chained_df = chain_dataframes(links, "Stance")
ensembled_df = backward_pass(chained_df, len(links))
```

To run a chain ensemble for all length `3` permutations of our links run the `get_permutations` function.

```python
sim_results_df = get_permutations(links, names, 3, "Stance", n_trails=20, backward = True)
```

**Note:** We assume all datasets used in the simulation come from the labeling function. It's critical that the dataset columns are in the same order from the labeler. If you are using the llm_label.py script, then this is handled for you automatically.
