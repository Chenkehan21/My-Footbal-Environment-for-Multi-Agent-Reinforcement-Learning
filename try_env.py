from collections import namedtuple

Observation = namedtuple('Observation', ['team', 'agent_id', 'obs'])
Actions = namedtuple('Actions', ['team', 'agent_id', 'actions'])
Rewards = namedtuple('Rewards', ['team', 'agent_id', 'rewards'])