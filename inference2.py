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

from collections import OrderedDict, defaultdict
from operator import itemgetter
from itertools import combinations

from model import TichuNet2, TichuNet3b
from tichu_env import *
from ai import RandomPlayer, GreedyPlayer, NeuralPlayer
from UI import *

def get_known_cards(game, observer):
    return set(game.players[observer].card_locs.keys())

def brier_calculator(odds, labels):
    assert type(odds) == list
    assert type(labels) == list
    se = [(o-l)**2 for o, l in zip(odds, labels)]
    return np.mean(se)

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

def model(game, observer, action, tmp_player):
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
    ai_player = tmp_player(game, game.turn)
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
    parser.add_argument('--player', '-p', type=str, default="greedy")
    args = parser.parse_args()
    my_id = args.observer

    if args.player == 'random':
        tmp_player = lambda g, i: RandomPlayer(g, i)
        players = [RandomPlayer for _ in range(4)]
    if args.player == 'greedy':
        tmp_player = lambda g, i: GreedyPlayer(g, i)
        players = [GreedyPlayer for _ in range(4)]
    elif args.player == 'neural':
        device = 'cpu'
        tn1 = TichuNet3b()
        if args.model:
            tn1.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
        #tn1.to(device)
        tmp_player = lambda g, i: NeuralPlayer(g, i, device=device, default_temp=.1)
        players = [lambda x, y: NeuralPlayer(x, y, tn1, device=device, default_temp=.1)
                   for _ in range(4)]
    game = Game(0, players)
    game.run_game(upto='firstAction')

    encoder = OneHotEncoder(game.deck.cards)

    idx_to_id = idx_2_id(my_id)
    id_to_idx = id_2_idx(my_id)
    global_card_dist = {
        card: torch.tensor([1./3, 1./3, 1./3])
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

            known_hand_cards = set()
            known_card_locs = game.players[my_id].card_locs
            for card in known_card_locs:
                if known_card_locs[card] == game.turn and card in game.unused_cards:
                    known_hand_cards.add(card)
            if real_action:
                known_hand_cards.update(real_action.cards)
            unknown_cards = set(game.unused_cards) - set(known_card_locs.keys()) - set(known_hand_cards)
            unknown_cards = list(unknown_cards)

            real_guess_cards = list(set(real_hand) - set(known_card_locs.keys()) - set(known_hand_cards))
            uniform_prior = len(real_guess_cards)/len(unknown_cards)

            cards_weight = {c: [] for c in unknown_cards}
            turn_idx = id_to_idx[game.turn]
            n_steps = 1000
            X = []
            y = []

            all_weights = []
            # do gradient steps
            for step in range(n_steps):
                cards, weight = model(game, my_id, real_action, tmp_player)
                X.append(encoder.encode(cards))
                y.append(weight)
                all_weights.append(weight)
                for card in cards_weight.keys():
                    if card in cards:
                        cards_weight[card].append(weight)
                    else:
                        cards_weight[card].append(0)
                if (step-1) % 100 == 0:
                    print("Step: {}".format(step))
            weight_sum = sum(all_weights)
            sorted_cards = sorted([(np.sum(cards_weight[card])/weight_sum, card) for card in cards_weight])
            print('----')
            print(f'Prior: {coloring(uniform_prior, bcolors.WARNING)}')
            print("="*50)
            updated_cards = []
            for weight, card in sorted_cards:
                print(coloring(card, bcolors.OKGREEN) if card in real_hand else str(card), weight)
                global_card_dist[card][turn_idx] *= weight/uniform_prior
                updated_cards.append(((weight / uniform_prior) * (global_card_dist[card][turn_idx]), card))
            true_labels = [int(c in real_hand) for _, c in sorted_cards]
            label_odds = [w for w, _ in sorted_cards]
            uniform_score = brier_calculator([uniform_prior]*len(true_labels), true_labels)
            print(f'               Uniform Brier score: {uniform_score}')
            single_action_score = brier_calculator(label_odds, true_labels)
            eval_color = bcolors.FAIL if single_action_score > uniform_score else bcolors.OKGREEN
            print(f'Single-action IS-using Brier score: {coloring(single_action_score, eval_color)}')

            print("="*50)
            updated_cards.sort()
            for weight, card in updated_cards:
                print(coloring(card, bcolors.OKGREEN) if card in real_hand else str(card), weight)
            print("="*50)
            label_odds = [w for w, _ in updated_cards]
            global_score = brier_calculator(label_odds, true_labels)
            eval_color = bcolors.FAIL if global_score > uniform_score else bcolors.OKGREEN
            print(f'        Uniform Brier score: {uniform_score}')
            print(f'Global IS-using Brier score: {coloring(global_score, eval_color)}')
        game.play(game.turn, real_action)
