""" hf_link.py

Python module for integrating opensource huggingface models into the the chain.

"""

from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration
from torch.nn.functional import softmax

MODELS_TESTED = {
    "google/flan-ul2": T5ForConditionalGeneration,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": AutoModelForCausalLM,
    "microsoft/Phi-3-medium-128k-instruct": AutoModelForCausalLM,
    "mistralai/Mistral-7B-Instruct-v0.2": AutoModelForCausalLM
}

class HuggingFaceLink:
    """

    A LLM chain link for huggingface models.

    Attributes:
        - model_name (str): The huggingface model.
        - model_class (callable): AutoModelForCausalLM or other like 
                                  T5ForConditionalGeneration.
        - labels (List[str]): A list of the zero-shot labels used for classification.
        - hf_token (str): The users huggingface token if neccesary.
        - quantization_config (BitsAndBytesConfig):
        
        - _model (AutoModel): Model loaded from AutoModel.from_pretrained()
        - _tokenizer (Tokenizer): Tokenizer loaded from Tokenizer.from_pretrained()
        - _label_token_ids (List[int]): List of token IDs associated with each label

    """
    
    def __init__(self, model_name, model_class, labels, 
                 hf_token = None, quantization_config = None):
        """
        Initializes a HuggingFaceLink

        """
        # Defined class attributes
        self.model_name = model_name
        self.model_class = model_class
        self.hf_token = hf_token
        self.labels = labels
        self.quantization_config = quantization_config

        self._model = None
        self._tokenizer = None
        self._label_token_ids = None

        # Just give user a simple warning message if they use a model that
        # is not tested.
        if self.model_name not in MODELS_TESTED.keys():
            print(f"[i] The model {self.model_name} is not tested and may have" \
                  " unexpected results during tokenization/inference.")

        if self.model_class not in MODELS_TESTED.values():
            print(f"[i] The model class {self.model_class.__name__} is not " \
                  "tested and may have unexpected results during tokenization/inference.")

    def load_model(self):
        """ Load model from Hugginface with the specified quantization.

        Args: None.
        Returns: None. Sets self._model and self._tokenizer.
        """
        try:
            self._model = self.model_class.from_pretrained(self.model_name,
                                                           token = self.hf_token,
                                                           device_map = "auto",
                                                           quantization_config = self.quantization_config
                                                          )
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Get the single token indicator associated with each label
            self._label_token_ids = []
            for label in self.labels:
                token_ids = self._tokenizer.encode(label, add_special_tokens=False)
                self._label_token_ids.append(token_ids[0])

        except ValueError as err:
            print(f"[!] Error loading model! {err}.")
            self._model = None
            self._tokenizer = None
        
    def unload_model(self):
        """ Clear memory to make space for new model to evaluate.

        Args: None.
        Returns: None. Clears models and tokenizer from memory.
        """
        del self._model
        del self._tokenizer
        torch.cuda.empty_cache()
        self._model = None
        self._tokenizer = None

    def get_labels(self, prompts, CoT = False, verbose = True,
                   max_response_len = 200):
        """
        Retreive the labels to a set of prompts.

        Args:
            - prompts (List[str] | List[Tuple[str]]): A list of prompts.
            - verbose (bool): Boolean indicator to make tqdm progress bar.
            - CoT (bool): Boolean indicator to perform CoT prompting.
            - max_response_len (int): The maximum number of tokens to return for
                CoT response if CoT is True.

        """
        if self._model is None or self._tokenizer is None:
            print("[!] Cannot get labels without loading model!")
            return None

        data_out = defaultdict(list)

        for prompt in tqdm(prompts, disable=(not verbose)):
            if CoT:
                label_logprobs, raw_resp, cot_resp = self._label_one_CoT(prompt,
                                                               max_response_len)
                data_out["CoT_resp"].append(cot_resp)

            else:
                label_logprobs, raw_resp = self._label_one(prompt)

            data_out["label_logprobs"].append(label_logprobs.tolist())
            data_out["pred_label"].append(self.labels[label_logprobs.argmax()])
            data_out["raw_pred_label"].append(raw_resp)

            top_logprobs = torch.topk(label_logprobs, 2).values
            conf_score = top_logprobs[0].item() - top_logprobs[1].item()
            data_out["conf_score"].append(conf_score)

        return dict(data_out)
        
    def _label_one(self, example):
        """
        Private method to label one example using the LLM.

        Args:
            - example (str): The single example to prompt the LLM with.
            - label_tokens (List[int])

        """
        tokenized_example = self._tokenize_example(example)
        
        input_ids = tokenized_example["input_ids"].to("cuda")
        attention_mask = tokenized_example["attention_mask"].to("cuda")

        model_out = self._model.generate(input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        max_new_tokens = 1,
                                        output_logits = True,
                                        return_dict_in_generate = True,
                                        pad_token_id = self._tokenizer.eos_token_id)
        
        raw_label = self._get_text_response(model_out, input_ids[0].shape[0])
        logprobs = torch.log(softmax(model_out.logits[0][0], dim=0))
        label_logprobs = logprobs[self._label_token_ids]
        
        return label_logprobs, raw_label
    
    def _label_one_CoT(self, prompt_pair, max_response_len):
        """
        Private method to label one observation using CoT prompting.
        """
        tokenized_example = self._tokenize_example(prompt_pair[0])
        
        input_ids = tokenized_example["input_ids"].to("cuda")
        attention_mask = tokenized_example["attention_mask"].to("cuda")
        
        model_out = self._model.generate(input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        max_new_tokens = max_response_len,
                                        output_logits = True,
                                        return_dict_in_generate = True,
                                        pad_token_id = self._tokenizer.eos_token_id)

        cot_resp = self._get_text_response(model_out, input_ids[0].shape[0])

        tokenzied_example = self._tokenize_example(prompt_pair, 
                                                   cot_response = cot_resp)

        input_ids = tokenized_example["input_ids"].to("cuda")
        attention_mask = tokenized_example["attention_mask"].to("cuda")

        model_out = self._model.generate(input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        max_new_tokens = 1,
                                        output_logits = True,
                                        return_dict_in_generate = True,
                                        pad_token_id = self._tokenizer.eos_token_id)
        
        raw_label = self._get_text_response(model_out, input_ids[0].shape[0])
        # The 10e-5 is a tiny addiditve factor to keepout -inf
        logprobs = torch.log(softmax(model_out.logits[0][0] + 10e-5, dim=0))
        label_logprobs = logprobs[self._label_token_ids]
        
        return label_logprobs, raw_label, cot_resp

    def _tokenize_example(self, example, cot_response = None):
        """ A helper function to tokenize a given example based on the model
            class passed.

        Args:
            - example (str): The example to tokenize.
            - cot_response (str): If provided a CoT response then the function
                                 will automatically assume the example is a 
                                 tuple and attempt to create a chat.

        Returns:
            - (dict): Dictionary with tokenized example and attention mask.
        """
        # AutoModelForCausalLM usually use the Chat Templates!
        if self.model_class is AutoModelForCausalLM:
            if cot_response is None:
                message = [{"role": "user", "content": example}]

            else: 
                message = [{"role": "user", "content": example[0]},
                            {"role": "assistant", "content": cot_response},
                            {"role": "user", "content": example[1]}]

            return self._tokenizer.apply_chat_template(message,
                                                       return_dict = True,
                                                       add_generation_prompt = True,
                                                       return_tensors = "pt")

        else:
            if cot_response is not None:
                example = example[0] + "\n\n" + cot_response + "\n\n" + example[1]
    
            return self._tokenizer(example, return_tensors="pt")

    def _get_text_response(self, model_out, input_len):
        """ A helper function to return the model's response in plain-text.

        Args:
            - model_out : The response from the model.generate() call.
            - input_len : The number of tokens used to prompt the model.
        
        """
        # AutoModelForCausalLM typically return all text including the input!
        if self.model_class is AutoModelForCausalLM:
            model_out_tokens = model_out["sequences"][0][input_len:]
            return self._tokenizer.decode(model_out_tokens, 
                                          skip_special_tokens = True)

        else:
            return self._tokenizer.decode(model_out.sequences[0], 
                                          skip_special_tokens = True)
