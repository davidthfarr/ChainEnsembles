""" stance_prompting.py

Module to build prompts for zero-shot stance classification.
A lot of this is based on Cruickshank and Ng "Prompting and Fine-Tuning 
Open-Sourced Language Models for Stance Classification," and Ziems et al. 
"Can Large Language Models Transform Computational Social Science?"

We follow their naming convention for prompting techniques.

"""

from .prompter import Prompter

class StancePrompter(Prompter):
    """ Defines functionality for all stance detection prompts.

    """

    def __init__(self, labels, column_map):
        """ Initializes the StancePrompter

        Args:
            - labels (list[str]): The list of labels to 
            - column_map (dict[str:str]): Maps the text to the dictionary key or
                column name. Does the same for the Target.
        """
        super().__init__(labels)
        self.column_map = column_map

        self.context_str = "Stance classification is the task of determining " \
                           "the expressed or implied opinion, or stance, of " \
                           "a statement toward a specific target. "

    def simple(self, **kwargs):
        """
        Function to build a simple zero-shot classification prompt from [Paper].
        Args:
            target (str): The target or topic for stance classification.
            statement (str): The statement we wish to analyze for stance.
        Returns:
            prompt (str): A string used to prompt an LLM
        """
        text = kwargs[self.column_map["text"]]
        target = kwargs[self.column_map["target"]]

        prompt = f'Statement: {text}\n\n' \
                 "Which of the following best describes the above social media " \
                 f"statements' stance regarding the idea of {target}? \n"
        prompt += self._make_multiple_choice_str()
        prompt += self._make_constraint_str()
        return prompt

    def CoT(self, **kwargs):
        """
        Function to build a chain-of-thought prompt. This is a zero-shot chain-of 
        thought prompt. There are two prompts
    
        Args:
            target (str): The target or topic for stance classification.
            statement (str): The statement we wish to analyze for stance.
    
        Returns:
            prompts (Tuple[str]): The strings used to prompt the model. The first
            is the question to analyze and the next is the question to classify the
            statement.
        """
        raise NotImplementedError
