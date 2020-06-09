import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions.constraints as constraints
import random
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
from UI import *
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from tichu_env import *
from ai import RandomPlayer, GreedyPlayer
from collections import OrderedDict
from operator import itemgetter

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
    known_cards = get_known_cards(game, observer)
    if action:
        known_cards.update(action.cards)
    unknown_cards = list(set(game.unused_cards) - known_cards)

    idx_to_id = idx_2_id(observer)
    id_to_idx = id_2_idx(observer)

    num_cards_in_hand = {
        i: len(set(game.players[idx_to_id[i]].hand) - known_cards)
        for i in idx_to_id
    }

    probs = []
    for card in unknown_cards:
        """
        FIXME: prior distribution of cards?
        """
        theta = [
            torch.tensor(global_card_dist[card][i])
            for i in idx_to_id
        ]
        player_probs = pyro.sample('{}_probs'.format(card), dist.Dirichlet(torch.stack(theta)))
        normalized_player_probs = player_probs #/ torch.sum(player_probs)
        probs.append(normalized_player_probs)

    probs = torch.stack(probs)
    hands = {i: list() for i in idx_to_id}
    card_probs = {i: list() for i in idx_to_id}
    for i, card in random.sample(tuple(enumerate(unknown_cards)), len(unknown_cards)):
        assigned = False
        while not assigned:
            player = torch.distributions.Categorical(probs=probs[i]).sample()
            #player = pyro.sample('{}_locs'.format(card), dist.Categorical(probs=probs[i]))
            if len(hands[int(player)]) < num_cards_in_hand[int(player)]:
                hands[int(player)].append(card)
                card_probs[int(player)].append(probs[i][int(player)])
                assigned = True
    """
    for i in idx_to_id:
        pyro.sample(
            '{}_card_assignment'.format(i),
            dist.Bernoulli(probs=torch.prod(torch.tensor(card_probs[i]))),
            obs=torch.tensor(1.)
        )
    """

    ai_player = GreedyPlayer(game, game.turn)
    ai_player.hand = hands[id_to_idx[game.turn]]
    if action:
        ai_player.hand += action.cards
    #print("===============card===============")
    #for card in ai_player.hand:
    #    print(card)
    #print("==================================")
    actions, action_probs = tuple([list(t) for t in zip(*ai_player.action_probs())])
    #print(actions, action_probs)
    #print(actions, action_probs)
    #print([str(a) for a in actions])
    #print(action)
    action_dist = dist.Categorical(probs=torch.tensor(action_probs))
    pyro.sample('action', action_dist, obs=torch.tensor(actions.index(action)))
    return hands

def guide(game, observer, action):
    if game.turn == observer:
        return None
    known_cards = get_known_cards(game, observer)
    if action:
        known_cards.update(action.cards)
    unknown_cards = list(set(game.unused_cards) - known_cards)

    idx_to_id = idx_2_id(observer)

    for card in unknown_cards:
        theta = [
            pyro.param("{}_{}".format(i, card), torch.tensor(global_card_dist[card][i]), constraint=constraints.positive)
            for i in idx_to_id
        ]
        player_probs = pyro.sample('{}_probs'.format(card), dist.Dirichlet(torch.stack(theta)))

my_id = 0
game = Game(0, [GreedyPlayer for i in range(4)])
scaler = 10.0
idx_to_id = idx_2_id(my_id)
global_card_dist = {
    card: [1.0/(NUM_PLAYERS-1) * scaler for i in range(NUM_PLAYERS - 1)]
    for card in game.deck.cards
}

game.run_game(upto='firstAction')

card_holders = {}
colors = [bcolors.FAIL, bcolors.OKGREEN, bcolors.OKBLUE]
for i in idx_to_id:
    for card in game.players[idx_to_id[i]].hand:
        card_holders[card] = i

print(card_holders)

for t in range(20):
    real_action = max(game.players[game.turn].action_probs(), key=itemgetter(1))[0]
    print(game.turn, real_action)

    if game.turn != my_id:
        # setup the optimizer
        adam_params = {"lr": 0.0001, "betas": (0.90, 0.999)}
        optimizer = Adam(adam_params)
        # setup the inference algorithm
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        n_steps = 1000
        # do gradient steps
        for step in range(n_steps):
            svi.step(game, my_id, real_action)
            if step % 100 == 0:
                print("Step: {}".format(step))

        print(game.turn)
        for card in game.players[game.turn].hand:
            print(card)

        for card in get_unknown_cards(game, my_id):
            if real_action and card in real_action.cards:
                continue
            # grab the learned variational parameters
            for i in idx_to_id:
                global_card_dist[card][i] = pyro.param("{}_{}".format(i, card)).item()
            for i in idx_to_id:
                global_card_dist[card][i] = global_card_dist[card][i]/sum(global_card_dist[card])*scaler
            print(card,
                " ".join([
                    coloring(str(global_card_dist[card][i]), colors[i])
                    if card_holders[card] == i
                    else str(global_card_dist[card][i])
                    for i in idx_to_id
                ])
            )
            #print(global_card_dist[card], "*****" if card in game.players[game.turn].hand else "")

    game.play(game.turn, real_action)
