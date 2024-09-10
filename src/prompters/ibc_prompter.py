""" ideology_prompter.py

A prompter for the ideology datasets.

"""

from .prompter import Prompter

class IBCPrompter(Prompter):
    """ Prompter for ideology detection.
    
    """
    
    def __init__(self, labels, column_map):
        super().__init__(labels)
        self.column_map = column_map

    def simple(self, **kwargs):
        """ Returns simple prompt."""
        text = kwargs[self.column_map["text"]]

        prompt = f'Statement: {text} \n'
        prompt += "Does the above political statement lean "
        prompt += self._make_class_list()
        prompt += "? Answer with only one word."

        return prompt
    
    def CoT(self, **kwargs):
        """ Returns a CoT prompt tuple."""
        raise NotImplementedError
