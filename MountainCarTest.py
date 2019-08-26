import gym

env = gym.make('MountainCar-v0')

print('action_space = ', env.action_space)
print('observation_space = ', env.observation_space)

for i_episode in range(1):
    
    observation = env.reset()
    
    for t in range(10):
    
        env.render()
        print('observation = ', observation, type(observation))
        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)

        print('action = ', action, type(action))
        print('reward = ', reward)
    	

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()