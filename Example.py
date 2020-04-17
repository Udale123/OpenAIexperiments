import gym
import NeuralNetwork as nn

brain = nn.NeuralNetwork([4,2,1])

env = gym.make('MountainCar-v0')
for i_episode in range(3):
    observation = env.reset()
    for t in range(100):
        env.render()
        #action = round(brain.FeedForward(observation)[0])
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

env = gym.make('MountainCar-v0')
print(env.action_space)
print(env.observation_space)