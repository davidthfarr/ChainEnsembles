"""hf_link.py

Python module for integrating opensource huggingface models into the the chain.

"""

import torch
from tqdm import tqdm
from collections import defaultdict
from typing import List, Tuple, Dict
from torch.nn.functional import softmax
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    QuantoConfig
)

MODELS_TESTED = {
    "google/flan-ul2": T5ForConditionalGeneration,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": AutoModelForCausalLM,
    "microsoft/Phi-3-medium-128k-instruct": AutoModelForCausalLM,
    "mistralai/Mistral-7B-Instruct-v0.2": AutoModelForCausalLM,
}

class HuggingFaceLink:
    """

    A LLM chain link for huggingface models.

    Attributes:
        model_name (str): The huggingface model.
        model_class (AutoModelForCausalLM | T5ForConditionalGeneration):
        labels (List[str]): A list of the target class labels for classification.
        hf_token (str): The users huggingface token if necessary.
        quantization_config (QuantoConfig): The config used for
            quantization for Huggingface models.

        _model (AutoModel): Model loaded from AutoModel.from_pretrained()
        _tokenizer (Tokenizer): Tokenizer loaded from Tokenizer.from_pretrained()
        _label_token_ids (List[int]): List of token IDs associated with each label.
        _device (str): Device architecture for loading models.

    """

    def __init__(
        self,
        model_name: str,
        model_class: AutoModelForCausalLM | T5ForConditionalGeneration,
        labels: List[str],
        hf_token: str = None,
        quantization_config: QuantoConfig = None,
    ):
        """Initializes a HuggingFaceLink
        
        Args:
            model_name (str):
            model_class (AutoModelForCausalLM | T5ForConditionalGeneratio):
            labels (List[str]):
            hf_token (str):
            quantization_config (QuantoConfig):

        Returns:
            
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
        self._set_device()

        # Warns user if using model that is untested
        if self.model_name not in MODELS_TESTED.keys():
            print(
                f"[i] The model {self.model_name} is not tested and may have",
                "unexpected results during tokenization/inference.",
            )

        if self.model_class not in MODELS_TESTED.values():
            print(
                f"[i] The model class {self.model_class.__name__} is not"
                "tested and may have unexpected results during tokenization/inference."
            )

    def load_model(self):
        """Load model from Huggingface with the specified quantization.

        Sets class attributes _model and _tokenizer.

        Args: None
        Returns: None
        """
        try:
            self._model = self.model_class.from_pretrained(
                self.model_name,
                token=self.hf_token,
                device_map="auto",
                quantization_config=self.quantization_config,
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
        """Clear memory and set _model and _tokenizer to None.

        Args: None
        Returns: None
        """
        del self._model
        del self._tokenizer
        torch.cuda.empty_cache()
        self._model = None
        self._tokenizer = None

    def get_labels(
        self,
        prompts: List[str] | List[Tuple[str]],
        CoT: bool = False,
        verbose: bool = True,
        max_response_len: int = 200,
    ) -> Dict:
        """Retrieve the labels to a set of prompts.

        Args:
            prompts (List[str] | List[Tuple[str]]): A list of prompts. If using
                CoT prompting, then expects prompts as list of Tuples.
            CoT (bool): Boolean indicator to perform CoT prompting.
            verbose (bool): Boolean indicator to make tqdm progress bar.
            max_response_len (int): The maximum number of tokens to return for
                CoT response if CoT is True.

        Returns:
            (dict): Data labeling data including confidence scores.
        """
        if self._model is None or self._tokenizer is None:
            print("[!] Cannot get labels without loading model!")
            return None

        data_out = defaultdict(list)

        for prompt in tqdm(prompts, disable=(not verbose)):
            if CoT:
                label_logprobs, raw_resp, cot_resp = self._label_one_CoT(
                    prompt, max_response_len
                )
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

    def _label_one(self, example: str) -> Tuple[torch.Tensor, str]:
        """
        Private method to label one example using the LLM.

        Args:
            example (str): The single example to prompt the LLM with.

        Returns:
            label_logprobs (torch.Tensor): A tensor with the label log
                probabilities.
            raw label (str): The raw token provided by the LLM response.
        """
        tokenized_example = self._tokenize_example(example)

        input_ids = tokenized_example["input_ids"].to(self._device)
        attention_mask = tokenized_example["attention_mask"].to(self._device)

        model_out = self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=self._tokenizer.eos_token_id,
            return_legacy_cache=True
        )

        raw_label = self._get_text_response(model_out, input_ids[0].shape[0])
        # The 10e-5 is a tiny additive factor to prevent -inf in softmax
        logprobs = torch.log(softmax(model_out.logits[0][0] + 10e-5, dim=0))
        label_logprobs = logprobs[self._label_token_ids]

        return label_logprobs, raw_label

    def _label_one_CoT(
        self, prompt_pair: List[Tuple[str, str]], max_response_len: int
    ) -> Tuple[torch.Tensor, str, str]:
        """Labels one observation using CoT prompting.

        Args:
            prompt_pair (List[Tuple[str, str]]): The pair of strings to form
                the CoT prompt.
            max_response_len (int): The maximum number of tokens to allow the
                LM from responding in CoT.

        Returns:
            label_logprobs (torch.Tensor): A tensor with the label log
                probabilities.
            raw label (str): The raw token provided by the LLM response.
            cot_resp (str): The LLMs CoT response.
        """
        tokenized_example = self._tokenize_example(prompt_pair[0])

        input_ids = tokenized_example["input_ids"].to(self._device)
        attention_mask = tokenized_example["attention_mask"].to(self._device)

        model_out = self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_response_len,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=self._tokenizer.eos_token_id,
            return_legacy_cache=True
        )

        cot_resp = self._get_text_response(model_out, input_ids[0].shape[0])

        tokenized_example = self._tokenize_example(prompt_pair, cot_response=cot_resp)

        input_ids = tokenized_example["input_ids"].to(self._device)
        attention_mask = tokenized_example["attention_mask"].to(self._device)

        model_out = self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=self._tokenizer.eos_token_id,
            return_legacy_cache=True
        )

        raw_label = self._get_text_response(model_out, input_ids[0].shape[0])
        # The 10e-5 is a tiny additive factor to prevent -inf in softmax
        logprobs = torch.log(softmax(model_out.logits[0][0] + 10e-5, dim=0))
        label_logprobs = logprobs[self._label_token_ids]

        return label_logprobs, raw_label, cot_resp

    def _tokenize_example(self, example, cot_response=None):
        """A helper function to tokenize a given example based on the model
            class passed.

        Args:
            example (str): The example to tokenize.
            cot_response (str): If provided a CoT response then the function
                will automatically assume the example is a
                tuple and attempt to create a chat.

        Returns:
            - (dict): Dictionary with tokenized example and attention mask.
        """
        if self.model_class is AutoModelForCausalLM:
            # AutoModelForCausalLM use the chat templates!
            if cot_response is None:
                message = [{"role": "user", "content": example}]

            else:
                message = [
                    {"role": "user", "content": example[0]},
                    {"role": "assistant", "content": cot_response},
                    {"role": "user", "content": example[1]},
                ]

            return self._tokenizer.apply_chat_template(
                message,
                return_dict=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

        else:
            if cot_response is not None:
                example = example[0] + "\n\n" + cot_response + "\n\n" + example[1]

            return self._tokenizer(example, return_tensors="pt")

    def _get_text_response(self, model_out: torch.LongTensor, input_len: str) -> str:
        """A helper function to return the model's response in plain-text.

        Args:
            model_out : The response from the model.generate() call as
                torch.LongTensor().
            input_len : The number of tokens used to prompt the model.

        Returns:
            (str): Raw text from LLM response.
        """
        if self.model_class is AutoModelForCausalLM:
            # AutoModelForCausalLM typically return all text including the input
            tokens = model_out["sequences"][0][input_len:]

        else:
            tokens = model_out.sequences[0]

        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    def _set_device(self):
        """Retrieves device architecture and sets _device.

        Args: None
        Returns: None
        """
        if torch.cuda.is_available():
            print("[i] Using CUDA.")
            self._device = "cuda"

        elif torch.backends.mps.is_available():
            print("[i] Using Apple Silicon.")
            self._device = "mps"

        else:
            print("[i] GPU not available, using CPU.")
            self._device = "cpu"
