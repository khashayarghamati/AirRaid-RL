import datetime
from pathlib import Path

from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, TransformObservation, FrameStack

from Config import Config
from agent import Agent
from environment import Environment
from metrics import MetricLogger

env = Environment().get_env()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=6)

checkpoint = None
agent = Agent(state_dim=6*84*84, action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

logger = MetricLogger(save_dir)

episodes = Config.total_episode

for e in range(episodes):

    state = env.reset()

    # Play the game!
    while True:
        action = agent.action(state[0])

        next_state, reward, done, truncated, info = env.step(action)

        q, loss = agent.learn(state=state[0], next_state=next_state, action=action, reward=reward)

        logger.log_step(reward, loss, q.cpu().detach().numpy())

        if done or truncated:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(
            episode=e,
            epsilon=agent.exploration_rate,
            step=agent.curr_step
        )
