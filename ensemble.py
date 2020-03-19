import numpy as np

from gating_network import GatingNetwork

class Simple_Ensemble(GatingNetwork):
    __metaclass__ = GatingNetwork

    def __init__(self, n_experts):
        """
            @param n_experts: Number of experts in expert net
        """
        super().__init__(n_experts)

    def get_weights(self):
        """
            Returns a np array of equal weights with shape [n_experts, 1]
        """
        return (1/self.n_experts) * np.ones([self.n_experts,1])