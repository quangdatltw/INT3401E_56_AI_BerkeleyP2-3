import itertools
from gridworld import Gridworld
from valueIterationAgents import ValueIterationAgent


def evaluate_parameters(discount_values, noise_values, living_reward_values, environment, grid, requirement):
    for discount, noise, living_reward in itertools.product(discount_values, noise_values, living_reward_values):
        # Initialize the environment with the required grid argument
        env = environment(grid)
        env.setNoise(noise)
        env.setLivingReward(living_reward)

        # Initialize the agent with the current parameters
        agent = ValueIterationAgent(env, discount=discount)

        # Run the evaluation 10 times
        pass_count = 0
        for _ in range(10):
            if evaluate_agent_performance(agent, env, requirement):
                pass_count += 1

        if pass_count == 10:
            print('(' + str(discount) + ', ' + str(noise) + ', ' + str(living_reward) + ')')
            '''print(discount, noise, living_reward)
            print("pass")
            print("----------------------")'''


def evaluate_agent_performance(agent, environment, requirement):
    desired_path = []

    if requirement == "Prefer the close exit (+1), risking the cliff (-10)":
        desired_path = ['east', 'east', 'north', 'exit']
    elif requirement == "Prefer the close exit (+1), but avoiding the cliff (-10)":
        desired_path = ['north', 'north', 'north', 'east', 'east', 'south', 'south', 'exit']
    elif requirement == "Prefer the distant exit (+10), risking the cliff (-10)":
        desired_path = ['east', 'east', 'east', 'east', 'north', 'exit']
    elif requirement == "Prefer the distant exit (+10), avoiding the cliff (-10)":
        desired_path = ['north', 'north', 'north', 'east', 'east', 'east', 'east', 'south', 'south', 'exit']

    return evaluate_policy(agent, environment, desired_path)


def evaluate_policy(agent, environment, desired_path, max_steps=20):
    state = environment.getStartState()
    steps = 0
    path = []

    while not environment.isTerminal(state) and steps < max_steps:
        action = agent.getAction(state)
        path.append(action)
        transitions = environment.getTransitionStatesAndProbs(state, action)
        next_state, prob = max(transitions, key=lambda x: x[1])
        state = next_state
        steps += 1

    return path == desired_path


# Define the ranges for discount, noise, and living reward values
discount_values = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
noise_values = [0, 0.05, 0.1]
living_reward_values = [-2, -1, -0.5, 0, 0.5, 1, 2]

# Define the environment, grid, and requirement
environment = Gridworld
grid = [
    ['_', '_', '_', '_', '_'],
    ['_', '#', '_', '_', '_'],
    ['_', '#', 1, '#', 10],
    ['S', '_', '_', '_', '_'],
    [-10, -10, -10, -10, -10]
]
requirement = "Prefer the close exit (+1), risking the cliff (-10)"
# requirement = "Prefer the close exit (+1), but avoiding the cliff (-10)"
# requirement = "Prefer the distant exit (+10), risking the cliff (-10)"
# requirement = "Prefer the distant exit (+10), avoiding the cliff (-10)"
evaluate_parameters(discount_values, noise_values, living_reward_values, environment, grid, requirement)
