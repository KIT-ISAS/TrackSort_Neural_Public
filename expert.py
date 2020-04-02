"""Abstract Expert and Expert Type.

Todo:
    * 
"""

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum, auto

class Expert(ABC):
    """The abstract expert class is a framework for every prediction expert.

    An expert has the basic attributes "type" and "name" to identify it.
    It can be useful to execute different functionality on different types of experts 
    (i.e. RNN training needs different handling than MLP training).

    Every expert should be able to perform basic predictions.

    Attributes:
        type (Expert_Type): The type of the expert
        name (String):      The name of the expert (For evaluation and logging)
    """

    def __init__(self, expert_type, name, model_path = ""):
        """Initialize a new expert.
        
        Args:
            type (Expert_Type): The type of the expert
            name (String):      The name of the expert (For evaluation and logging)
            model_path (String):The path to save or load the model
        """
        self.type = expert_type
        self.name = name
        self.model_path = model_path

    @abstractmethod
    def train_batch(self, inp, target):
        """Train expert on a batch of training data.
        
        Must return predictions for the input data even if no training is done.

        Args:
            inp (tf.Tensor):    A batch of input tracks
            target (tf.Tensor): A batch of tagets to predict to

        Returns:
            predictions (np.array or tf.Tensor): Predicted positions for training instances
        """
        pass

    @abstractmethod
    def predict_batch(self, inp):
        """Predict a batch of input data for testing.

        Args:
            inp (tf.Tensor): A batch of input tracks

        Returns
            prediction (np.array or tf.Tensor): Predicted positions for training instances
        """
        pass

    @abstractmethod
    def save_model(self):
        """Save the model to its model path."""
        pass

    @abstractmethod
    def load_model(self):
        """Load the model from its model path."""
        pass

    def get_type(self):
        """Return type."""
        return self.type
        
class Expert_Type(Enum):
    """Simple enumeration class for expert types."""
    
    RNN = auto()
    KF = auto()
    MLP = auto()