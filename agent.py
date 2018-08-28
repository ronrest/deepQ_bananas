# %load_ext autoreload
# %autoreload 2
import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim

# SETTINGS
EXPERIENCE_MEMORY_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """ Creates an agent that interacts with a Unity-ML Environment
        using a Deep Q-learning model (in pytorch).
    """
    def __init__(self, n_state, n_actions, n_hidden=32, n_layers=2, seed=333, snapshotfile="snapshot.pth"):
        """ Initialize the agent.

        Args:
            n_state     (int):  Number of features that represent the state
            n_actions   (int):  Number of actions available to agent
            n_hidden    (int):  Number of units in hidden neural net layers
            n_layers    (int):  Number of layers for neural network
            seed        (int):  Set the random seed (for reproducibility)
            snapshotfile (str): Filepath to use for saving weights
        """
        self.n_state = n_state
        self.n_actions = n_actions
        self.seed = random.seed(seed)
        self.snapshotfile = snapshotfile

        # Deep Q-Network
        self.qnetwork_local = QNetwork(n_state, n_actions, seed, n_hidden=64).to(device)
        self.qnetwork_target = QNetwork(n_state, n_actions, seed, n_hidden=64).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.loss_func = torch.nn.MSELoss(reduce=True)

        # Experience Replay Memory
        self.memory = ReplayBuffer(n_actions, EXPERIENCE_MEMORY_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # TODO: have the is_training attribute control eval and train
        #       mode in pytprch network
        self.is_training = True

    def memorize_and_learn_step(self, state, action, reward, next_state, done):
        """ Given  S,A,R',S' and if it is finished, it saves the eperience
            to memory, and occasionally samples from memorized experiences and
            learns from those memories.
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Once every UPDATE_EVERY steps, randomly sample memories to learn from
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def choose_action(self, state, epsilon=0.0):
        """ Given an environment state, it returns an action using epsilon
            greedy policy.

        Args:
            state   (array_like): current state
            epsilon (float)     : probability of choosing a random action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad(): # temporarially set requires_grad flag to false
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.n_actions))

    def learn(self, experiences, gamma):
        """ Update the weights of the neural network representing the Q values,
            given a batch of experience tuples.

        Args:
            experiences (tuple of torch.Variable): tuple with the following
                        torch tensors
                        (states, actions, rewards, next_states, dones)
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Q_TARGET
        next_logits = self.qnetwork_target(next_states).detach() # no need to calculate gradients, so detach
        q_next = torch.max(next_logits, dim=1, keepdim=True)[0]
        # where dones=1, it  will ignore q_next, and just use current reward
        q_target = rewards + ((1-dones)*(gamma * q_next))

        # Q_CURRENT - based on action taken in experience
        current_logits = self.qnetwork_local(states)
        q_pred = torch.gather(current_logits, 1, actions)

        # LOSS
        loss = self.loss_func(q_pred, q_target)
        # loss = F.mse_loss(q_pred, q_target)

        # OPTIMIZE WEIGHTS
        self.optimizer.zero_grad() # zero the parameter gradients
        loss.backward()
        self.optimizer.step()

        # UPDATE TARGET NETWORK
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """ Performs a soft update on the target Q network weights, by
            shifting them slightly towards the local Q network by a factor of
            `tau`.

            θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model  (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def snapshot(self, file=None):
        """ Takes a snapshot file of the neural netowrk weights """
        file = self.snapshotfile if file is None else file
        torch.save(self.qnetwork_local.state_dict(), file)

    def load_snapshot(self, file=None):
        """ Loads the neural network weights from a file """
        file = self.snapshotfile if file is None else file
        self.qnetwork_local.load_state_dict(torch.load(file))
        self.qnetwork_target.load_state_dict(torch.load(file))



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, n_actions, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            n_actions (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.n_actions = n_actions
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
