import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """ A neural network that represents the Q-action values for a
        reinforcement learning agent.

        Two fully connected hidden layers, with ReLu activations, followed by
        a fully connected output layer.
    """
    def __init__(self, state_size, action_size, seed=333, n_hidden=32):
        """ Initialize the weights and build the components of the model.

        Args:
            state_size  (int): Shape of the state (as the input layer size)
            action_size (int): Number of actions (as the output layer size)
            seed        (int): Random seed, for reproducibility
            n_hidden    (int): Number of nodes in the hidden layers.
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(in_features=state_size, out_features=n_hidden, bias=True)
        self.fc2 = nn.Linear(in_features=n_hidden, out_features=n_hidden, bias=True)
        self.fc_out = nn.Linear(in_features=n_hidden, out_features=action_size, bias=False)

    def forward(self, state):
        """ Builds the forward pass structure of the network, which maps from
            state to action values
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x
