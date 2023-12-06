
import gymnasium as gym

from Config import Config


class Environment:

    def __init__(self):
        self.env = gym.make(Config.env_name, render_mode='rgb_array')
        self.env.reset()

    def get_env(self):
        return self.env
