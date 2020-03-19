from abc import ABC, abstractmethod

class GatingNetwork(ABC):

    def __init__(self, n_experts):
        """
            @param n_experts: Number of experts in expert net
        """
        self.n_experts = n_experts

    @abstractmethod
    def get_weights(self, **kwargs):
        """
            Returns the weights for all experts.
        """
        pass

