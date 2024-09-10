""" misinfo_prompter.py

Module to build prompts for zero-shot misinformation classification.
A lot of this is based on Cruickshank and Ng "Prompting and Fine-Tuning 
Open-Sourced Language Models for Stance Classification," and Ziems et al. 
"Can Large Language Models Transform Computational Social Science?"

We follow their naming convention for prompting techniques.

"""

from .prompter import Prompter 

class MisinfoPrompter(Prompter):
    """ Prompter for misinformation detection.
    
    """

    def __init__(self, labels, column_map):
        super().__init__(labels)
        self.column_map = column_map


    def simple(self, **kwargs):
        """ Returns simple prompt. """
        text = kwargs[self.column_map["text"]]
        
        prompt = f"Headline: {text}\n\nIs the above news headline more likely to be "
        prompt += self._make_class_list()
        prompt += "? Answer with only one word."
        
        return prompt
    
    def CoT(self):
        """ Returns a CoT prompt tuple. """
        raise NotImplementedError