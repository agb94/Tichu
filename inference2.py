import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions.constraints as constraints
import random
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

#from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale
from scipy.stats import rankdata

from collections import OrderedDict
from operator import itemgetter
from itertools import combinations

from model import TichuNet2, TichuNet3b
from tichu_env import *
from ai import RandomPlayer, GreedyPlayer, NeuralPlayer
from UI import *

def get_known_cards(game, observer):
    return set(game.players[observer].card_locs.keys())

def get_unknown_cards(game, observer):
    return list(set(game.unused_cards) - get_known_cards(game, observer))

def idx_2_id(observer):
    others = [pid for pid in range(NUM_PLAYERS) if pid != observer]
    mapping = OrderedDict()
    for i, pid in enumerate(others):
        mapping[i] = pid
    return mapping

def id_2_idx(observer):
    others = [pid for pid in range(NUM_PLAYERS) if pid != observer]
    mapping = OrderedDict()
    for i, pid in enumerate(others):
        mapping[pid] = i
    return mapping

def model(game, observer, action):
    if game.turn == observer:
        return None
    known_hand_cards = set()
    known_card_locs = game.players[observer].card_locs
    for card in known_card_locs:
        if known_card_locs[card] == game.turn and card in game.unused_cards:
            known_hand_cards.add(card)
    if action:
        known_hand_cards.update(action.cards)
    unknown_cards = list(set(game.unused_cards) - set(known_card_locs.keys()) - set(known_hand_cards))

    idx_to_id = idx_2_id(observer)
    id_to_idx = id_2_idx(observer)

    num_cards_in_hand = len(set(game.players[game.turn].hand)) - len(known_hand_cards)
    sampled = []
    turn_idx = id_to_idx[game.turn]
    card_dist = [1.0] * len(unknown_cards)
    for i in range(num_cards_in_hand):
        picked = pyro.sample("card_{}".format(i), dist.Categorical(probs=torch.tensor(card_dist)))
        sampled.append(unknown_cards[picked])
        card_dist[picked] = torch.tensor(.0)
    ai_player = NeuralPlayer(game, game.turn, tn1, device=device)
    ai_player.hand = sampled[:]
    ai_player.hand += list(known_hand_cards)

    actions, action_probs = tuple([list(t) for t in zip(*ai_player.action_probs())])
    if action in actions:
        action_dist = dist.Categorical(probs=torch.tensor(action_probs))
        return sampled, action_probs[actions.index(action)]
    else:
        return sampled, 0.0

class OneHotEncoder:
    def __init__(self, cards):
        self.cards = list(cards)

    def encode(self, cards):
        return [int(card in cards) for card in self.cards]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--observer', '-o', type=int, default=None)
    args = parser.parse_args()
    my_id = args.observer
    device = 'cpu'
    tn1 = TichuNet3b()
    if args.model:
        tn1.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    #tn1.to(device)
    robot_players = [lambda x, y: NeuralPlayer(x, y, tn1, device=device, default_temp=.1)
                     for _ in range(4)]
    game = Game(0, robot_players)
    game.run_game(upto='firstAction')

    encoder = OneHotEncoder(game.deck.cards)

    idx_to_id = idx_2_id(my_id)
    id_to_idx = id_2_idx(my_id)
    global_card_dist = {
        card: torch.tensor([1., 1., 1.])
        for card in game.deck.cards
    }


    card_holders = {}
    colors = [bcolors.FAIL, bcolors.OKGREEN, bcolors.OKBLUE]
    for i in idx_to_id:
        for card in game.players[idx_to_id[i]].hand:
            card_holders[card] = i

    print(card_holders)

    for t in range(20):
        real_action = game.players[game.turn].sample_action()
        print([str(a) for a in game.current], game.turn, real_action)
        real_hand = game.players[game.turn].hand
        if game.turn != my_id:
            cards_weight = {}
            turn_idx = id_to_idx[game.turn]
            n_steps = 1000
            X = []
            y = []
            # do gradient steps
            for step in range(n_steps):
                cards, weight = model(game, my_id, real_action)
                X.append(encoder.encode(cards))
                y.append(weight)
                for card in cards:
                    if card not in cards_weight:
                        cards_weight[card] = list()
                    cards_weight[card].append(weight)
                if (step-1) % 10 == 0:
                    print("Step: {}".format(step))
                #print(cards, weight)
            sorted_cards = sorted([(np.mean(cards_weight[card]), card) for card in cards_weight])
            print("="*50)
            updated_cards = []
            for weight, card in sorted_cards:
                print(coloring(card, bcolors.OKGREEN) if card in real_hand else str(card), weight)
                global_card_dist[card][turn_idx] *= weight
                updated_cards.append((weight * global_card_dist[card][turn_idx], card))
            print("="*50)
            updated_cards.sort()
            for weight, card in updated_cards:
                print(coloring(card, bcolors.OKGREEN) if card in real_hand else str(card), weight)
            print("="*50)
        game.play(game.turn, real_action)
