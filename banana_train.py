"""
Trains an agent to interact with the Banana environment, and saves a snapshot
of the trained network once it has solved the environment.
"""
# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
# %matplotlib inline
from environment import UnityEnvWrapper, Interaction
from agent import Agent


# ##############################################################################
#                                                          SETUP THE ENVIRONMENT
# ##############################################################################
# SETUP ENVIRONMENT, AGENT AND INTERACTION OBJECTS
seed = 344
env = UnityEnvWrapper("Banana_Linux/Banana.x86_64", seed=seed)

state = env.reset()
state_size = env.state_size
n_actions = env.n_actions

agent = Agent(n_state=state_size, n_actions=n_actions, n_layers=2, n_hidden=64, seed=seed)
interaction = Interaction(env=env, agent=agent, solved_score=13, solved_window=100)

# TRAIN
print("\nTRAINING AGENT\n")
interaction.interact_and_learn(n_episodes=2000, max_steps=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995)

# PLOT THE SCORES OVER TIME
fig, ax = plt.subplots(figsize=(11, 6))
fig.suptitle('Average score over time (averaged over 100 episodes)', fontsize=15, fontdict={"fontweight": "extra bold"})
ax.plot(interaction.train_scores, linewidth=1.0, color='#307EC7', linestyle='-')
ax.set_xlabel("Episode Number")
ax.set_ylabel("Average Score")
fig.savefig("traincurve.png")

# INTERACT WITH ENVIRONMENT - in evaluation mode
# interaction.interact(max_steps=1000, verbose=True)

# EXAMPLE PRINTOUT OF TRAINING
#
# Episode 100	Average Score: 0.572
# Episode 200	Average Score: 2.57
# Episode 300	Average Score: 6.70
# Episode 400	Average Score: 9.97
# Episode 500	Average Score: 12.46
# Episode 519	Average Score: 13.02
# Environment solved in 419 episodes!	Average Score: 13.02
