from conway_env import ConwayEnv, FlatObservationWrapper, FlatActionWrapper
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

plt.ion()


def simple_test():
    env = ConwayEnv()
    check_env(env)
    print(env.observation_space.n)


def run_test():
    env = ConwayEnv()
    plt.figure()
    img_plot = plt.imshow(env.state, interpolation="nearest", cmap=plt.cm.gray)
    plt.show(block=False)
    for i in range(10000):
        action = env.action_space.sample()
        rrr = env.step(action)
        img_plot.set_data(rrr[0])
        plt.draw()
        plt.pause(0.1)
        if rrr[2]:
            break
    # print(rrr)


def sb3_test():
    env = FlatActionWrapper(FlatObservationWrapper(ConwayEnv()))
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100, log_interval=4)

    env = FlatActionWrapper(FlatObservationWrapper(ConwayEnv()))
    obs = env.reset()
    plt.figure()
    obs_im = obs.reshape((8, 8))
    print(obs_im)
    img_plot = plt.imshow(obs_im, interpolation="nearest", cmap=plt.cm.gray)
    plt.show(block=False)
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        obs_im = obs.reshape((8, 8))
        img_plot.set_data(obs_im)
        plt.draw()
        plt.pause(0.5)
        if done:
            # break
            obs = env.reset()


if __name__ == '__main__':
    sb3_test()
