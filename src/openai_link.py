""" openai_link.py

A module that defines a LLM link using the OpenAI API and langchain!

"""

import torch
from langchain_openai import ChatOpenAI

from tqdm import tqdm
from collections import defaultdict

class OpenAILink:
    """
    A LLM chain link for huggingface models.

    Attributes:
        - model_name (str): The OpenAI model.
        - labels (List[str]): A list of the zero-shot labels used for classification.
        
        - _openai_token (str): The OpenAI API key. Optional argument. Langchain
                               will automatically check the enviromental variable
                               OPENAI_API_KEY for the key when initializing model.
        - _chat (ChatOpenAI): OpenAI chat model via langchain for simple
                                 question answering.
        - _cot_chat (ChatOpenAI): OpenAI chat model via langchain for doing CoT
                                  analysis.

    """
    def __init__(self, model_name, labels, openai_token = None):
        """
        Initializes a OpenAILink

        """
        self.model_name = model_name
        self.labels = labels

        self._openai_token = openai_token
        self._chat = None
        self._cot_chat = None
        self._label_token_ids = None

    def load_model(self):
        """ Ensure valid connection to OpenAI API and load required models.

        Args: None.
        Returns: None. Sets self._model and self._tokenizer.
        """
        self._chat = ChatOpenAI(model = self.model_name,
                                max_tokens = 1,
                                temperature = 0,
                                logprobs = True,
                                top_logprobs=20,
                                api_key = self._openai_token)
        
        self._cot_chat = ChatOpenAI(model = self.model_name,
                                    temperature = 0,
                                    api_key = self._openai_token)

        # Get the token indicator associated with each label
        self._label_token_ids = []
        for label in self.labels:
            self._label_token_ids.append(self._chat.get_token_ids(label)[0])
        
    def unload_model(self):
        """ Reset Open AI chat to None.

        Args: None.
        Returns: None.
        """
        self._chat = None
        self._cot_chat = None

    def get_labels(self, prompts, CoT = False, verbose = True,
                   max_response_len = None, logit_bias = True):
        """
        Retreive the labels to a set of prompts.

        Args:
            - prompts (List[str] | List[Tuple[str]]): A list of prompts.
            - verbose (bool): Boolean indicator to make tqdm progress bar.
            - CoT (bool): Boolean indicator to perform CoT prompting.
            - max_response_len (int): The maximum number of tokens to return for
                CoT response if CoT is True.

        """
        if self._chat is None or (CoT is True and self._cot_chat is None):
            print("[!] Cannot get labels without loading model!")
            return None

        data_out = defaultdict(list)

        for prompt in tqdm(prompts, disable=(not verbose)):
            if CoT:
                self._cot_chat = self._cot_chat.bind(max_tokens = max_response_len)
                label_logprobs, raw_resp, cot_resp = self._label_one_CoT(prompt)
                data_out["CoT_resp"].append(cot_resp)

            else:
                if logit_bias is True:
                    logit_bias_dict = {tid: 10 for tid in self._label_token_ids}
                    self._chat = self._chat.bind(logit_bias = logit_bias_dict)

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
            - label_token_ids (List[int]): A list of label token IDs.

        """
        messages = [("human", example)]
        chat_resp = self._chat.invoke(messages)

        label_logprobs = self._get_label_logprobs(chat_resp)
        
        return label_logprobs, chat_resp.content
    
    def _label_one_CoT(self, prompt_pair):
        """
        Private method to label one observation using CoT prompting.
        """
        messages = [("human", prompt_pair[0])]
        cot_resp = self._cot_chat.invoke(messages)
        
        messages = [("human", prompt_pair[0]),
                    ("assistant", cot_resp.content),
                    ("human", prompt_pair[1])]
        chat_resp = self._chat.invoke(messages)

        label_logprobs = self._get_label_logprobs(chat_resp)
        
        return label_logprobs, chat_resp.content, cot_resp.content

    def _get_label_logprobs(self, chat_response):
        """ Helper function to retreive the log probability for each label given
        a chat response.

        Args:
            chat_response (AImessage): The response from an invoke call.

        Returns:
            (torch.Tensor): A tensor of log probabilites for each label.
        
        """
        token_probs = chat_response.response_metadata['logprobs']['content'][0]['top_logprobs']

        # DNE value is the min token logprob minus 1.
        dne_val = token_probs[len(token_probs) - 1]['logprob'] - 1
        results = {token_id: dne_val for token_id in self._label_token_ids}

        for entry in token_probs:
            token_id = self._chat.get_token_ids(entry['token'])[0]
            if token_id in self._label_token_ids:
                results[token_id] = entry['logprob']

        return torch.tensor(list(results.values()))

