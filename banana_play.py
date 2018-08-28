"""
Make a trained agent interact with the world using its internal policy
"""
from environment import UnityEnvWrapper, Interaction
from agent import Agent

seed = 76543
env = UnityEnvWrapper("Banana_Linux/Banana.x86_64", seed=seed)
state = env.reset()
state_size = env.state_size
n_actions = env.n_actions

agent = Agent(n_state=state_size, n_actions=n_actions, n_layers=2, n_hidden=64, seed=seed)
agent.load_snapshot()
interaction = Interaction(env=env, agent=agent, solved_score=13, solved_window=100)
interaction.interact(max_steps=1000, verbose=True)
