import matplotlib.pyplot as plt
import numpy as np
import torch

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

from tichu_env import *
"""
def weather():
    cloudy = pyro.sample('cloudy', pyro.distributions.Bernoulli(0.3))
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
    scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
    temp = pyro.sample('temp', pyro.distributions.Normal(mean_temp, scale_temp))
    return cloudy, temp.item()

for _ in range(3):
    print(weather())

def scale(guess):
    weight = pyro.sample("weight", dist.Normal(guess, 1.0))
    return pyro.sample("measurement", dist.Normal(weight, 0.75))

def deferred_conditioned_scale(measurement, guess):
    return pyro.condition(scale, data={"measurement": measurement})(guess)

conditioned_scale = pyro.condition(scale, data={"measurement": 9.5})
"""

"""
global variable card_dist should be like

{
    card('Black', 2) : { 0: 0.5, 1: 0.2, 2: 0.3, 3: 0.0 }
    ...
    ...
}
"""

game = Game(seed=0)
my_id = 1
game.mark_exchange(0, 1, 0)
game.mark_exchange(0, 2, 7)
game.mark_exchange(0, 3, 1)
game.mark_exchange(1, 0, 0)
game.mark_exchange(1, 2, 1)
game.mark_exchange(1, 3, 7)
game.mark_exchange(2, 0, 7)
game.mark_exchange(2, 1, 0)
game.mark_exchange(2, 3, 1)
game.mark_exchange(3, 0, 0)
game.mark_exchange(3, 1, 7)
game.mark_exchange(3, 2, 1)
print(game.exchange_index)
game.exchange()

print(game.players[my_id].card_locs)

def card_dist(game, player_id):
    card_dist = {} # should be updated during the game
    card_locs = game.players[player_id].card_locs
    for card in game.unused_cards:
        if card in card_locs:
            dist = {i: 1. if i == card_locs[card] else 0. for i in range(NUM_PLAYERS)}
        else:
            dist = {i: 1/(NUM_PLAYERS-1) if i != player_id else 0. for i in range(NUM_PLAYERS)}
        card_dist[card] = dist
    return card_dist
print(card_dist(game, my_id))

def hand(card_dist):
    card = pyro.sample('card', pyro.distributions.Categorical(0, len(cards)))
    print(card)
