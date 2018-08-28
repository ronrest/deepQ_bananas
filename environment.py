import sys
import numpy as np
from collections import deque
from unityagents import UnityEnvironment


class UnityEnvWrapper(object):
    """ An object that wraps around the unity environment, returning an
        environment object that has a simplified API interface, similar,
        but not exactly the same as OpenAI's Gym environment.

        You can access the underlying unity environment object using:

            myobj.env
    """
    def __init__(self, envfile, train_mode=True, seed=333, **kwargs):
        """ Given a filepath to a Unity environment file, it initializes the
            wrapper object.

        Args:
            envfile     (str):  filepath to Unity environment file
            train_mode  (bool): Set up the environment in training mode?
            seed        (int):  Random seed, for reproducibility
            **kwargs:   aditional keyword arguments to pass to UnityEnvironment
        """
        np.random.seed(seed)
        self.env = UnityEnvironment(file_name=envfile,
                                    seed=seed,
                                    **kwargs)

        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # reset environment, and get info from perspective of one of the brains
        # self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.env_info = None
        self.train_mode = train_mode

    @property
    def n_actions(self):
        return self.brain.vector_action_space_size

    @property
    def state_size(self):
        return len(self.env_info.vector_observations[0])

    @property
    def s(self):
        """ Returns the state of the environment """
        try:
            return self.env_info.vector_observations[0]
        except:
            return None

    @property
    def state(self):
        """ Returns the state of the environment """
        return self.s

    def set_train_mode(self, mode):
        self.train_mode = mode

    def reset(self):
        """ Reset the environment, and returns the new state """
        self.env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        return self.s

    def step(self, action):
        """ Perform an action in the environment, and returns a tuple with the
            following:

            next_state, reward, done, info

            Where `info` is a UnityEnvironment env_info object.
        """
        self.env_info = self.env.step(action)[self.brain_name] # perform action
        # next_state = self.env_info.vector_observations[0]    # get next state
        next_state = self.s                                    # get next state
        reward = self.env_info.rewards[0]                      # get reward
        done = self.env_info.local_done[0]                 # is episode is over?
        return next_state, reward, done, self.env_info

    def sample(self):
        """ Randomly chose and return an action using uniform distribution """
        return np.random.randint(self.n_actions)



class VisualUnityEnvWrapper(UnityEnvWrapper):
    """ Wrapper for Unity environments that have images as states """
    def __init__(self, envfile, train_mode=True, seed=333, **kwargs):
        UnityEnvWrapper.__init__(self, envfile, train_mode=True, seed=333, **kwargs)

    @property
    def state_size(self):
        stateinfo = self.brain.camera_resolutions[0]
        color_channels = 1 if stateinfo["blackAndWhite"] else 3
        state_size = [stateinfo["height"], stateinfo["width"], color_channels]
        return state_size

    @property
    def s(self):
        try:
            # Assumes that by default Unity environment returns a batchsize of 1
            # images, and shape [1, H, W, C]
            return self.env_info.visual_observations[0][0]
        except:
            return None


class Interaction(object):
    """ Object that handles the interaction between an agent and an environment
    """
    def __init__(self, env, agent, solved_score=100, solved_window=100):
        """ Initialize the Interaction object.

        Args:
            env:    Environment object
            agent:  Agent object
            solved_score  (float): min value needed to consider problem solved
            solved_window (int)  : Number of consecutive episodes where the
                                   solved score needs to be averaged over in
                                   order to be considered solved.
        """
        self.env = env
        self.agent = agent
        self.solved_score = solved_score
        self.solved_window = solved_window
        self.train_scores = [] # keeps track of the scores during training
        self.scores_window = deque(maxlen=solved_window)  # last solved_window scores

    def interact_and_learn(self, n_episodes, max_steps=1000,
                           eps_start=1.0, eps_end=0.01, eps_decay=0.995,
                           ):
        """ Makes an agent intereact with the world and learn from its interactions.

        Args:
            n_episodes  (int):   max number of training episodes
            max_steps   (int):   max number of timesteps per episode
            eps_start   (float): starting value of epsilon, for epsilon-greedy action
                                 selection
            eps_end     (float): min value of epsilon
            eps_decay   (float): Decay rate of epsilon for eac episode.

            snapshot_file   (str):   filepath to the saved weights
        """
        # Backup agent and environment settings
        env_train_mode_bu = self.env.train_mode
        agent_training_mode_bu = self.agent.is_training

        # Set agent and environment for training mode
        self.env.set_train_mode(True)
        self.agent.is_training = True

        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            state = self.env.reset()
            score = 0
            for t in range(max_steps):
                action = self.agent.choose_action(state, epsilon=eps)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.memorize_and_learn_step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            self.scores_window.append(score)       # save most recent score
            self.train_scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon

            # Check if environment has been solved yet
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(self.scores_window)), end="")
            if i_episode % self.solved_window == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(self.scores_window)))
            if np.mean(self.scores_window)>=self.solved_score:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(self.scores_window)))
                self.agent.snapshot()
                break

        # restore agent and environment settings
        self.env.set_train_mode(env_train_mode_bu)
        self.agent.is_training = agent_training_mode_bu


    def interact(self, max_steps=1000, verbose=True):
        """ Makes the agent interact with the environment using its internal
            policy for making actions.

            Returns the score at the end of running.
        """
        # Backup agent and environment settings
        train_mode_bu = self.env.train_mode # backup current train_mode
        agent_training_mode_bu = self.agent.is_training

        # Set agent and environment settings for evaluation mode
        self.env.set_train_mode(False)
        self.agent.is_training = False

        score = 0
        state = self.env.reset()
        for t in range(max_steps):
            action = self.agent.choose_action(state, epsilon=0)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            score += reward
            if done:
                break

            # Feedback of reward
            if verbose:
                print('\rSTEP: {:04d} TOTAL SCORE: {:02.2f}  Current Reward {:02.2f}'.format(t, score, reward, end=""))
                sys.stdout.flush()

            # Check if it has reached termimal state
            if score >= self.solved_score:
                if verbose:
                    print("SUCESS!!!")
                break

        # restore agent and environment settings
        self.env.set_train_mode(train_mode_bu) # restore previous train_mode
        self.agent.is_training = agent_training_mode_bu
        return score
