import gym
from gym import spaces, logger
import math
import numpy as np
from gym.utils import seeding
from copy import deepcopy
import warnings
import os

class TaskT(gym.Env):
    metadata = {'name':'TaskT', 'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
    spec = {'id':'TaskT'}
    def __init__(self, sections=1, seq='RGB', final_reward=False, reward_obs=True, R=4, saving=False, log_dir="./TaskT_log/"):
        """
        Sequential target reaching task.
        :param sections: how many targets to reach to finish the task
        :param seq: any combination of 'R', 'G', 'B' to indicated the required sequence of target-reaching.
        :param final_reward: if True, only final target provides reward, otherwise all targets provide reward.
        :param reward_obs: whether reward is one element of observation
        :param R: difficulty (distance between targets)
        :param saving: whether to save steps/rewards into txt file
        :param log_dir: directory to save steps/rewards
        """
        self.sections = sections
        self.saving = saving
        self.log_dir = log_dir
        self.final_reward = final_reward
        self.reward_obs = reward_obs
        self.sequence = seq
        self.R = R
        self.reward = 0.0
        self.reward_signal = 0.0
        self.dim_position = 2
        self.dim_action = 2
        self.speed = 0.8
        self.radius = 0.5
        self.max_steps = 128
        self.steps = 0

        self.init_position = np.array([7.5, 7.5], dtype=np.float32)
        self.init_position[0] += np.float32(15 * (np.random.rand() - 0.5))
        self.init_position[1] += np.float32(15 * (np.random.rand() - 0.5))
        self.old_position = self.init_position
        self.new_position = self.init_position
        self.orientation = 2 * np.pi * np.random.rand()

        self.init_state = 0
        self.size = 1
        self.action_space = spaces.Box(low=-1., high=1., shape=(2,))
        if reward_obs:
            self.observation_space = spaces.Box(low=-1., high=5., shape=(12,))
        else:
            self.observation_space = spaces.Box(low=-1., high=1., shape=(11,))

        self.reward_range = (-np.Inf, np.Inf)

        self._seed()

        if self.saving:
            if os.path.exists(log_dir):
                warnings.warn('{} exists (possibly so do data).'.format(log_dir))
            else:
                os.makedirs(log_dir)

            path = self.log_dir + 'TaskT' + '.txt'
            self.file_pointer = open(path, 'w+')

        self.red_position = np.float32(R * (np.random.rand(self.dim_position) - 0.5)) + np.array([7.5, 7.5],
                                                                                                 dtype=np.float32)
        while True:
            self.green_position = np.float32(R * (np.random.rand(self.dim_position) - 0.5)) + np.array([7.5, 7.5],
                                                                                                       dtype=np.float32)
            if (np.sum((self.red_position - self.green_position) ** 2)) > 2:
                break
        while True:
            self.blue_position = np.float32(R * (np.random.rand(self.dim_position) - 0.5)) + np.array([7.5, 7.5],
                                                                                                      dtype=np.float32)
            if (np.sum((self.blue_position - self.green_position) ** 2)) > 2 and (np.sum((self.blue_position - self.red_position) ** 2)) > 2:
                break

        self.first_experience = 0
        self.second_experience = 0
        self.third_experience = 0

        if self.sequence[0] == 'R':
            self.first_position = self.red_position
        elif self.sequence[0] == 'G':
            self.first_position = self.green_position
        elif self.sequence[0] == 'B':
            self.first_position = self.blue_position

        if self.sections >= 2:
            if self.sequence[1] == 'R':
                self.second_position = self.red_position
            elif self.sequence[1] == 'G':
                self.second_position = self.green_position
            elif self.sequence[1] == 'B':
                self.second_position = self.blue_position

        if self.sections >= 3:
            if self.sequence[2] == 'R':
                self.third_position = self.red_position
            elif self.sequence[2] == 'G':
                self.third_position = self.green_position
            elif self.sequence[2] == 'B':
                self.third_position = self.blue_position

        self.done = 0

        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.reward = 0.0
        self.steps = 0
        self.reward_signal = 0.0
        self.init_position = np.array([7.5, 7.5], dtype=np.float32)
        self.init_position[0] += np.float32(15 * (np.random.rand() - 0.5))
        self.init_position[1] += np.float32(15 * (np.random.rand() - 0.5))
        self.old_position = self.init_position
        self.new_position = self.init_position
        self.orientation = 2 * np.pi * np.random.rand()

        R = self.R
        self.red_position = np.float32(R * (np.random.rand(self.dim_position) - 0.5)) + np.array([7.5, 7.5],
                                                                                                 dtype=np.float32)
        while True:
            self.green_position = np.float32(R * (np.random.rand(self.dim_position) - 0.5)) + np.array([7.5, 7.5],
                                                                                                       dtype=np.float32)
            if (np.sum((self.red_position - self.green_position) ** 2)) > 2:
                break
        while True:
            self.blue_position = np.float32(R * (np.random.rand(self.dim_position) - 0.5)) + np.array([7.5, 7.5],
                                                                                                      dtype=np.float32)
            if (np.sum((self.blue_position - self.green_position) ** 2)) > 2 and (np.sum((self.blue_position - self.red_position) ** 2)) > 2:
                break

        self.first_experience = 0
        self.second_experience = 0
        self.third_experience = 0

        if self.sequence[0] == 'R':
            self.first_position = self.red_position
        elif self.sequence[0] == 'G':
            self.first_position = self.green_position
        elif self.sequence[0] == 'B':
            self.first_position = self.blue_position

        if self.sections >= 2:
            if self.sequence[1] == 'R':
                self.second_position = self.red_position
            elif self.sequence[1] == 'G':
                self.second_position = self.green_position
            elif self.sequence[1] == 'B':
                self.second_position = self.blue_position

        if self.sections >= 3:
            if self.sequence[2] == 'R':
                self.third_position = self.red_position
            elif self.sequence[2] == 'G':
                self.third_position = self.green_position
            elif self.sequence[2] == 'B':
                self.third_position = self.blue_position

        self.done = 0
        return self.get_obs()

    def get_obs(self):

        lambd = 3.0

        position = self.new_position
        theta = self.orientation

        red_dis = np.sqrt(np.sum((position - self.red_position) ** 2))
        green_dis = np.sqrt(np.sum((position - self.green_position) ** 2))
        blue_dis = np.sqrt(np.sum((position - self.blue_position) ** 2))

        if 0 <= theta and theta < np.pi / 2:
            dw1 = min((15 - position[1]) / abs(np.sin(theta)), (15 - position[0]) / abs(np.cos(theta)))
            dw2 = min((position[1] - 0) / abs(np.sin(theta)), (position[0] - 0) / abs(np.cos(theta)))
        elif np.pi / 2 <= theta and theta < np.pi:
            dw1 = min((15 - position[1]) / abs(np.sin(theta)), (position[0] - 0) / abs(np.cos(theta)))
            dw2 = min((position[1] - 0) / abs(np.sin(theta)), (15 - position[0]) / abs(np.cos(theta)))
        elif np.pi <= theta and theta < 3 * np.pi / 2:
            dw1 = min((position[1] - 0) / abs(np.sin(theta)), (position[0] - 0) / abs(np.cos(theta)))
            dw2 = min((15 - position[1]) / abs(np.sin(theta)), (15 - position[0]) / abs(np.cos(theta)))
        else:
            dw1 = min((position[1] - 0) / abs(np.sin(theta)), (15 - position[0]) / abs(np.cos(theta)))
            dw2 = min((15 - position[1]) / abs(np.sin(theta)), (position[0] - 0) / abs(np.cos(theta)))

        tr = np.arctan2(self.red_position[1] - position[1], self.red_position[0] - position[0]) - theta
        tg = np.arctan2(self.green_position[1] - position[1], self.green_position[0] - position[0]) - theta
        tb = np.arctan2(self.blue_position[1] - position[1], self.blue_position[0] - position[0]) - theta

        if self.reward_obs:
            obs = np.array([np.exp(-red_dis / lambd),
                            np.exp(-green_dis / lambd),
                            np.exp(-blue_dis / lambd),
                            np.exp(-dw1 / lambd),
                            np.exp(-dw2 / lambd),
                            np.sin(tr),
                            np.cos(tr),
                            np.sin(tg),
                            np.cos(tg),
                            np.sin(tb),
                            np.cos(tb),
                            self.reward_signal])
        else:
            obs = np.array([np.exp(-red_dis / lambd),
                            np.exp(-green_dis / lambd),
                            np.exp(-blue_dis / lambd),
                            np.exp(-dw1 / lambd),
                            np.exp(-dw2 / lambd),
                            np.sin(tr),
                            np.cos(tr),
                            np.sin(tg),
                            np.cos(tg),
                            np.sin(tb),
                            np.cos(tb)])
        return obs

    def get_init_position(self):
        return self.init_position

    def reward_fun(self):

        position = self.old_position
        new_position = self.new_position

        if new_position[0] > 15 or new_position[0] < 0 or new_position[1] > 15 or new_position[1] < 0:
            r = -0.1
        # self.done = 1
        else:
            if not self.first_experience:
                target_position = self.first_position
                dis2 = np.sum((new_position - target_position) ** 2)
                if dis2 < 0.16:
                    self.first_experience = 1
                    if self.sections == 1:
                        self.done = 1
                    r = 20.0 / (1 + np.sqrt(dis2)) if (self.sections == 1 or (not self.final_reward)) else 0.0
                else:
                    r = 0.0
            elif self.sections >= 2 and (not self.second_experience):
                target_position = self.second_position
                dis2 = np.sum((new_position - target_position) ** 2)
                if dis2 < 0.16:
                    self.second_experience = 1
                    if self.sections == 2:
                        self.done = 1
                    r = 50.0 / (1 + np.sqrt(dis2)) if (self.sections == 2 or (not self.final_reward)) else 0.0
                else:
                    r = 0.0
            elif self.sections >= 3:
                target_position = self.third_position
                dis2 = np.sum((new_position - target_position) ** 2)
                if dis2 < 0.16:
                    self.third_experience = 1
                    if self.sections == 3:
                        self.done = 1
                    r = 100.0 / (1 + np.sqrt(dis2)) if self.sections == 3 else 0.0
                else:
                    r = 0.0
            else:
                r = 0.0
        return r

    def step(self, action, saving=None):

        if self.done:
            warnings.warn("Task already done!! Not good to continue!")

        if saving is None:
            saving = self.saving

        self.old_position = deepcopy(self.new_position)
        self.steps += 1
        action = np.reshape(action, [self.dim_action])

        current_action_1 = action[0]
        current_action_2 = action[1]

        self.new_position[0] = self.old_position[0] + np.clip(
            self.speed * np.cos(self.orientation) * (current_action_1 + current_action_2) / 2, -self.speed, self.speed)
        self.new_position[1] = self.old_position[1] + np.clip(
            self.speed * np.sin(self.orientation) * (current_action_1 + current_action_2) / 2, -self.speed, self.speed)

        self.orientation += np.clip((current_action_1 - current_action_2) / 2 / self.radius, -np.pi,
                                    np.pi)
        while self.orientation >= 2 * np.pi:
            self.orientation -= 2 * np.pi
        while self.orientation < 0:
            self.orientation += 2 * np.pi

        self.reward = self.reward_fun()
        self.reward_signal = self.reward

        if self.new_position[0] > 15 or self.new_position[0] < 0 or self.new_position[1] > 15 or self.new_position[
            1] < 0:
            self.new_position = self.old_position

        if self.steps >= self.max_steps:
            self.done = 1

        if saving and self.saving:
            self.savelog(self.reward, self.done)

        return self.get_obs(), self.reward, self.done, {}

    def savelog(self, r, d):
        if self.saving:
            self.file_pointer.write("%f, %f \n" % (r, d))
        else:
            warnings.warn('cannot save!')

    def render(self, mode='human'):
        screen_width = 500
        screen_height = 500

        world_width = 15
        scale = screen_width / world_width
        # carty = 100 # TOP OF CART
        # polewidth = 10.0
        # polelen = scale * (2 * self.length)
        # cartwidth = 50.0
        # cartheight = 30.0
        target_radius = 0.3
        car_radius = 0.1

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(0, 15, 0, 15)
            self.red = rendering.make_circle(target_radius)
            self.green = rendering.make_circle(target_radius)
            self.blue = rendering.make_circle(target_radius)
            self.red.set_color(1., 0., 0.)
            self.green.set_color(0., 1., 0.)
            self.blue.set_color(0., 0., 1.)
            self.red_trans = rendering.Transform()
            self.green_trans = rendering.Transform()
            self.blue_trans = rendering.Transform()
            self.red.add_attr(self.red_trans)
            self.green.add_attr(self.green_trans)
            self.blue.add_attr(self.blue_trans)
            self.viewer.add_geom(self.red)
            self.viewer.add_geom(self.green)
            self.viewer.add_geom(self.blue)

            self.car = rendering.make_circle(car_radius)
            self.car.set_color(0., 0., 0.)
            self.car_trans = rendering.Transform()
            self.car.add_attr(self.car_trans)
            self.viewer.add_geom(self.car)

        self.red_trans.set_translation(self.red_position[0], self.red_position[1])
        self.green_trans.set_translation(self.green_position[0], self.green_position[1])
        self.blue_trans.set_translation(self.blue_position[0], self.blue_position[1])
        self.car_trans.set_translation(self.new_position[0], self.new_position[1])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))
