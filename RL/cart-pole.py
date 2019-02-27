import gym
import numpy as np
import math

np.random.seed(0)

env = gym.make('CartPole-v0')
env.seed(0)

actions = [i for i in range(env.action_space.n)]
num_of_actions = len(actions)

bins = [3, 1, 6, 3]
obs_high = env.observation_space.high
obs_high[[1, 3]] = [10,  math.radians(50)]
obs_low  = env.observation_space.low
obs_low[[1, 3]] = [-10,  -math.radians(50)]
num_of_states = np.prod(bins)

def find_bin(val, low, high, n_bins):
    val = min(max(low, val), high)
    for i in range(n_bins):
        if low + i * (high-low) / n_bins <= val and val <= low + (i+1)* (high-low) / n_bins:
            return i
    return None

def get_state(obs):
    global obs_low, obs_high, bins
    quantized_state = 0
    for i in range(len(obs)):
        quantized_state *= bins[i]
        quantized_state += find_bin(obs[i], obs_low[i], obs_high[i], bins[i])
    return quantized_state

class agent:
    def __init__(self, env, num_of_states, num_of_actions):
        self.env = env
        self.Q = np.zeros((num_of_states, num_of_actions))

    def greedy_action(self, s):
        return np.argmax(self.Q[s, :])

    def eps_greedy_action(self, eps, s):
        if np.random.random() < eps:
            return np.random.randint(0, num_of_actions)
        return np.argmax(self.Q[s, :])

    def update_q_values(self, s, a, r, new_s, gamma, alpha):
        self.Q[s, a] += alpha * (r + gamma * np.max(self.Q[new_s, :]) - self.Q[s, a])

    def train(self, episodes, alpha, gamma, eps, save=True):
        avg_reward = 0
        for episode in range(1, episodes+1):
            o = self.env.reset()
            s = get_state(o)
            t = 0
            is_done = False
            while not is_done:
                t += 1
                a = self.eps_greedy_action(eps, s)
                new_o, r, is_done, info = self.env.step(a)
                new_s = get_state(new_o)
                if is_done and t < self.env._max_episode_steps:
                    r = -400
                self.update_q_values(s, a, r, new_s, gamma, alpha)
                s = new_s
                avg_reward += r

            if episode % 10 == 0:
                eps *= 0.99
                alpha *= 0.99

            if episode % 100 == 0:
                print('average reward for episodes ({0}-{1}): {2}'\
                .format(episode-100, episode, avg_reward/100.))
                avg_reward = 0
        np.save('Q.npy', self.Q)

    def test(self, episodes, load=True):
        if load == True:
            try:
                self.Q = np.load('Q.npy')
            except:
                print('Q.npy not found. Train the model first!')
        for episode in range(1, episodes +1):
            total_reward = 0
            o = self.env.reset()
            s = get_state(o)
            is_done = False
            while not is_done:
                a = self.greedy_action(s)
                self.env.render()
                new_o, r, is_done, info = self.env.step(a)
                new_s = get_state(new_o)
                s = new_s
                total_reward += r
            print('reward for episodes {0}: {1}'\
                .format(episode, total_reward))
        self.env.close()

env._max_episode_steps = 300
EPISODES = 2000
ALPHA = 0.4
GAMMA = 0.99
EPS = 0.2

cart_pole_agent = agent(env, num_of_states, num_of_actions)
cart_pole_agent.train(EPISODES, ALPHA, GAMMA, EPS)
cart_pole_agent.test(10)
