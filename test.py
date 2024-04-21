import GameEnv
import pygame
import numpy as np
from saveload import save_network,load_network

from collections import deque
import random, math

TOTAL_GAMETIME = 10000
N_EPISODES = 10000
REPLACE_TARGET = 10

game = GameEnv.RacingEnv()
game.fps = 60

GameTime = 0
GameHistory = []
renderFlag = True



car = load_network("Car_best")

ddqn_scores = []
eps_history = []


def play(net):
    # scores = deque(maxlen=100)

    for e in range(N_EPISODES):
        # reset env
        game.reset()


        score = 0
        counter = 0

        gtime = 0
        renderFlag = True

        # first step
        observation_, reward, done = game.step(0)
        observation = np.array(observation_)

        while not done:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    return

            # new
            action = np.argmax(net.predict_single(observation))
            observation_, reward, done = game.step(action)
            observation_ = np.array(observation_)

            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward

            observation = observation_

            gtime += 1

            if gtime >= TOTAL_GAMETIME:
                done = True

            if renderFlag:
                game.render(action)

play(car)
