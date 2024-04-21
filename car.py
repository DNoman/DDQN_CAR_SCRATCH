import numpy as np
import gym
import random
from tqdm import tqdm
import pygame
from nn import NeuralNetwork
import copy
from collections import deque
import GameEnv
import pygame
import numpy as np
from collections import deque
import random, math
import matplotlib.pyplot as plt
from saveload import save_network,load_network


# def get_surface(rgb_array):
#     surface = pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))
#     return surface
#
#
# # utility function to view how our agent plays the cartpole, using pygame
# # After done; this function will print the score (total reward)
# def play(net, env):
#     pygame.init()
#     screen = pygame.display.set_mode((600, 400))
#     pygame.display.set_caption('CartPole')
#
#     state, _ = env.reset()
#     done = False
#     rewards = 0
#     while not done:
#         action = np.argmax(net.predict_single(state))
#         state, r, done, _, _ = env.step(action)
#         rewards += r
#         surface = get_surface(env.render())
#
#         screen.blit(surface, (0, 0))
#         pygame.display.flip()
#
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT: done = True
#
#     print(rewards)
#     pygame.quit()


def learn(data, batch_size):
    D.append(data)
    if len(D) < batch_size: return

    minibatch = random.sample(D, batch_size)
    X = np.zeros((batch_size, state_shape))
    y = np.zeros((batch_size, action_size))
    for i, (state, action, reward, nxt_state, done) in enumerate(minibatch):
        X[i] = state
        y_i = reward + (1 - done) * gamma * np.max(Q_target.predict_single(nxt_state))
        y[i] = Q.predict_single(state)
        y[i][action] = y_i
        # for Q(s_i,a_i) - y_i we let our network to compute Q(s_i,a_i),
        # so every index except the action became zero:- [0,si_ai-y_i,0,0])**2

    Q.train_on_batch(X, y, epoch=1)

def update_target(target):
    for i in range(Q.L):
        W1,b1 = Q.NN[i]
        W2,b2 = target[i]
        W2[:] = W1[:]
        b2[:] = b1[:]


def train(num_episode=100, batch_size=32, C=40, ep=10):
    global epsilon, best_score
    steps = 0
    for i in tqdm(range(1, num_episode + 1)):
        game.reset()
        episode_reward = 0
        episode_loss = 0
        counter = 0
        gtime = 0
        renderFlag = True
        # Sample Phase
        done = False
        nxt_state, reward, done = game.step(0)
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            state = np.array(nxt_state)
            # epsilon = min(epsilon_min, epsilon * epsilon_decay)  # e decay

            # e-greedy(Q)
            x = np.random.rand()
            if x < epsilon:
                action = np.random.randint(action_size)
            else:
                q_vals = Q.predict_single(state)
                action = np.argmax(q_vals)

            nxt_state, reward, done = game.step(action)
            nxt_state= np.array(nxt_state)
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0
            episode_reward += reward

            # Learining Phase
            learn((state, action, reward, nxt_state, int(done)), batch_size)
            # print("state : ",state)
            # print("next_state: ",nxt_state)
            gtime += 1
            if gtime >= TOTAL_GAMETIME:
                done = True
            if renderFlag:
                game.render(action)

            steps += 1

            if steps % C == 0: update_target(Q_target.NN)
            if epsilon < epsilon_min:
                epsilon = epsilon_min
            else:
                epsilon = epsilon * epsilon_decay
        if episode_reward >= 100:
            save_network(Q, "Car_more_than_100")
        if episode_reward >= 60:
            save_network(Q, "Car_more_than_60")
        if episode_reward >= 50:
            save_network(Q, "Car_more_than_50")
        print(f"Episode: {i} Reward: {episode_reward} epsilon: {epsilon}")
        eps_history.append(i)
        ddqn_scores.append(episode_reward)
        max_steps_history.append(episode_reward)
        avg_score = np.mean(ddqn_scores[max(0, i - 100):(i + 1)])
        if len(max_steps_history) >= 100:
            avg_steps = np.mean(max_steps_history[-100:])
            avg_steps_history.append(avg_steps)
        plt.plot(eps_history, ddqn_scores, marker='o', linestyle='-')
        if i >= 100:
            plt.plot(eps_history[100:], avg_steps_history[1:], label='Avg step', linestyle='--')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('BIỂU ĐỒ REWARD ĐẠT ĐƯỢC TRÊN MỖI EPISODE ')
        plt.legend(['Reward', 'Avg_step'])
        plt.grid(True)
        plt.pause(0.05)

ddqn_scores = []
eps_history = []
max_steps_history = []
avg_steps_history = []
game = GameEnv.RacingEnv()
game.fps = 60
TOTAL_GAMETIME = 10000000000

arch = [19,64,64,5]
af = ["leakyrelu","relu","linear"]
# Q
Q = NeuralNetwork(arch,af,eta=5e-4,momentum=0,seed=8)

# Q'
Q_target = copy.deepcopy(Q) #Q' NeuralNetwork(same parms as above) then update_target(Q_target.NN) will also work

# Replay Memory
D = deque(maxlen=20000) # if D==maxlen and we append new data oldest one will get removed

action_size = 5 # Action Space
state_shape = 19 # State Size

# Epsilon
epsilon = 1
epsilon_min = 0.01
epsilon_decay = 0.9995

# Gamma
gamma = 0.99
# Just to check the highest score obtained during training
best_score = -np.inf
train(20000,512,ep=100)
save_network(Q,"Car")
save_network(Q,"Car")
cpsnb = load_network("CartPoleScratchNetBetter")