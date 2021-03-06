import gym
import NeuralNetwork as nn
import numpy as np

# choose enviroment to learn about

env = gym.make('BipedalWalker-v3')
inputs = 24  # numperical inputs
outputs = 4  # one output to -1 or 1





# brain layers
layers = [inputs, int((inputs + outputs) / 2), outputs]

# simulation parameters
generation_population = 30
max_time_steps = 50
number_of_generations = 100
mutation_rate = 0.05

best_scores = np.zeros(number_of_generations)
best_performers = []

starting_population = []
index = []
for i in range(generation_population):
    starting_population.append(nn.NeuralNetwork(layers, activation='tanh'))
    index.append(i)

current_population = starting_population.copy()
current_scores = np.zeros(len(starting_population))

for gen in range(number_of_generations):
    best_score = -1000
    current_scores = np.zeros(len(starting_population))
    for pop in range(generation_population):
        observation = env.reset()
        for t in range(max_time_steps):
            env.render()
            action = (current_population[pop].feed_forward(observation))
            observation, reward, done, info = env.step(action)
            current_scores[pop] += reward
            if done:
                # print("Episode finished after {} timesteps".format(t + 1))
                break
        if max(current_scores) > best_score:
            best_score = max(current_scores)
    best_scores[gen] = sum(current_scores) / len(current_scores)

    # choose top half
    top_half = [x for _, x in sorted(zip(current_scores, index))][int(generation_population / 2):]
    keepers = [current_population[i] for i in top_half]
    for idx, value in enumerate(keepers):
        current_population[idx] = value
        child = value.cross_over(np.random.choice(keepers))
        child.mutate(mutation_rate)
        current_population[len(keepers) + idx] = child
    best_performers.append(keepers[-1])
    print(gen, best_score, best_scores[gen])
env.close()

print(best_scores)