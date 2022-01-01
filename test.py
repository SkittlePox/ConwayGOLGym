from conway_env import ConwayEnv, FlatObservationWrapper, FlatActionWrapper
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np

plt.ion()


def simple_test():
    env = ConwayEnv()
    check_env(env)
    print(env.observation_space.n)


def run_test():
    # env = ConwayEnv(state_shape=(16, 16), goal_location=(8, 8))
    env = ConwayEnv()
    env.reset()
    plt.figure()
    img_plot = plt.imshow(env.state, interpolation="nearest", cmap=plt.cm.gray)
    plt.show(block=False)
    for i in range(10000):
        action = env.action_space.sample()
        rrr = env.step(action)
        img_plot.set_data(rrr[0])
        plt.draw()
        plt.pause(0.5)
        if rrr[2]:
            # break
            env.reset()
    # print(rrr)


def evaluate(model, env, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)

        obs, reward, done, info = env.step(action)

        # Stats
        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward


def sb3_test():
    env = FlatActionWrapper(FlatObservationWrapper(ConwayEnv()))
    model = PPO("MlpPolicy", env, verbose=1)
    evaluate(model, env, num_steps=1000)
    model.learn(total_timesteps=10000, log_interval=4)

    evaluate(model, env, num_steps=1000)

    # env = FlatActionWrapper(FlatObservationWrapper(ConwayEnv()))
    # obs = env.reset()
    # plt.figure()
    # obs_im = obs.reshape((8, 8))
    # img_plot = plt.imshow(obs_im, interpolation="nearest", cmap=plt.cm.gray)
    # plt.show(block=False)
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     obs_im = obs.reshape((8, 8))
    #     img_plot.set_data(obs_im)
    #     plt.draw()
    #     plt.pause(0.5)
    #     if done:
    #         # break
    #         obs = env.reset()


if __name__ == '__main__':
    run_test()
