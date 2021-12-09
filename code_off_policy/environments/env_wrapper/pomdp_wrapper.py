import numpy as np
import gym

class POMDPWrapper(gym.ObservationWrapper):
    def __init__(self, env_name, random_sensor_missing_prob=0.1):

        super().__init__(gym.make(env_name))
        self.random_sensor_missing_prob = random_sensor_missing_prob

    def observation(self, obs):
        obs[np.random.rand(len(obs)) <= self.random_sensor_missing_prob] = 0
        return obs.flatten()
