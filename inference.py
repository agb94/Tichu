import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions.constraints as constraints
import random
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from tichu_env import *
from ai import RandomPlayer

def get_known_cards(game, observer):
    return set(game.players[observer].card_locs.keys())

def get_unknown_cards(game, observer):
    return set(game.unused_cards) - get_known_cards(game, observer)

def model(game, observer, action):
    if game.turn == observer:
        return None
    known_cards = get_known_cards(game, observer)
    unknown_cards = get_unknown_cards(game, observer)
    other_players, num_hand_cards = [], []
    for pid in range(NUM_PLAYERS):
        if pid == observer:
            continue
        other_players.append(pid)
        num_hand_cards.append(len(set(game.players[pid].hand) - known_cards))
    player_idx = other_players.index(game.turn)
    hand_probs = torch.tensor(num_hand_cards, dtype=torch.float32)
    assert torch.sum(hand_probs) == len(unknown_cards)
    normalized_hand_probs = hand_probs / torch.sum(hand_probs)
    #print("There are {}/{} unknown locations of cards".format(len(unknown_cards), len(game.deck.cards)))
    #print("Size of hands {}: {}".format(other_players, hand_probs))
    #print("probability of hands of {}: {}".format(other_players, normalized_hand_probs))

    unknown_cards = list(unknown_cards)
    probs = []
    for card in unknown_cards:
        """
        FIXME: prior distribution of cards?
        """
        a = torch.tensor(1.0)
        b = torch.tensor(1.0)
        c = torch.tensor(1.0)
        player_probs = pyro.sample('{}_probs'.format(card), dist.Dirichlet(torch.stack([a, b, c])))
        normalized_player_probs = player_probs #/ torch.sum(player_probs)
        #print(card, normalized_player_probs)
        probs.append(normalized_player_probs)
    probs = torch.stack(probs)

    hands = [list() for p in other_players]
    card_probs = [list() for p in other_players]
    for i, card in random.sample(tuple(enumerate(unknown_cards)), len(unknown_cards)):
        assigned = False
        while not assigned:
            player = torch.distributions.Categorical(probs=probs[i]).sample()
            #player = pyro.sample('{}_locs'.format(card), dist.Categorical(probs=probs[i]))
            if len(hands[int(player)]) < num_hand_cards[int(player)]:
                hands[int(player)].append(card)
                card_probs[int(player)].append(probs[i][int(player)])
                assigned = True

    for i, card in enumerate(hands[player_idx]):
        #print(card, card_probs[player_idx][i])
        pass

    ai_player = RandomPlayer(game, other_players[player_idx], hands[player_idx])

    actions, probs = tuple([list(t) for t in zip(*ai_player.action_probs())])
    if action not in actions:
        actions.append(action)
        probs.append(.0)

    action_dist = dist.Categorical(probs=torch.tensor(probs))
    pyro.sample('action', action_dist, obs=torch.tensor(actions.index(action)))
    return hands

def guide(game, observer, action):
    if game.turn == observer:
        return None
    known_cards = get_known_cards(game, observer)
    unknown_cards = get_unknown_cards(game, observer)
    other_players, num_hand_cards = [], []
    for pid in range(NUM_PLAYERS):
        if pid == observer:
            continue
        other_players.append(pid)
        num_hand_cards.append(len(set(game.players[pid].hand) - known_cards))
    player_idx = other_players.index(game.turn)
    hand_probs = torch.tensor(num_hand_cards, dtype=torch.float32)
    assert torch.sum(hand_probs) == len(unknown_cards)
    normalized_hand_probs = hand_probs / torch.sum(hand_probs)
    #print("There are {}/{} unknown locations of cards".format(len(unknown_cards), len(game.deck.cards)))
    #print("Size of hands {}: {}".format(other_players, hand_probs))
    #print("probability of hands of {}: {}".format(other_players, normalized_hand_probs))

    unknown_cards = list(unknown_cards)
    probs = []
    for card in unknown_cards:
        """
        FIXME: prior distribution of cards?
        """
        a = pyro.param("a_{}".format(card), torch.tensor(1.0), constraint=constraints.positive)
        b = pyro.param("b_{}".format(card), torch.tensor(1.0), constraint=constraints.positive)
        c = pyro.param("c_{}".format(card), torch.tensor(1.0), constraint=constraints.positive)
        player_probs = pyro.sample('{}_probs'.format(card), dist.Dirichlet(torch.stack([a, b, c])))
        normalized_player_probs = player_probs #/ torch.sum(player_probs)
        #print(card, normalized_player_probs)
        probs.append(normalized_player_probs)
    return
    """
    return
    probs = torch.stack(probs)

    hands = [list() for p in other_players]
    card_probs = [list() for p in other_players]
    for i, card in random.sample(tuple(enumerate(unknown_cards)), len(unknown_cards)):
        assigned = False
        while not assigned:
            #player = pyro.sample('{}_locs'.format(card), dist.Categorical(probs=probs[i]))
            player = torch.distributions.Categorical(probs=probs[i]).sample()
            if len(hands[int(player)]) < num_hand_cards[int(player)]:
                hands[int(player)].append(card)
                card_probs[int(player)].append(probs[i][int(player)])
                assigned = True

    for i, card in enumerate(hands[player_idx]):
        #print(card, card_probs[player_idx][i])
        pass

    ai_player = RandomPlayer(game, other_players[player_idx], hands[player_idx])

    actions, probs = tuple([list(t) for t in zip(*ai_player.action_probs())])
    if action not in actions:
        actions.append(action)
        probs.append(.0)

    action_dist = dist.Categorical(probs=torch.tensor(probs))
    pyro.sample('action', action_dist, obs=torch.tensor(actions.index(action)))
    return hands
    """

game = Game(seed=0)
my_id = 0
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
game.exchange()
game.play(game.turn, game.players[game.turn].possible_actions()[0])

# setup the optimizer
adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)
# setup the inference algorithm
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
n_steps = 1000
# do gradient steps
action = game.players[game.turn].possible_actions()[0]
for step in range(n_steps):
    svi.step(game, my_id, action)
    if step % 100 == 0:
        print("Step: {}".format(step))

print(game.turn)
for card in game.players[game.turn].hand:
    print(card)

for card in get_unknown_cards(game, my_id):
    # grab the learned variational parameters
    a = pyro.param("a_{}".format(card)).item()
    b = pyro.param("b_{}".format(card)).item()
    c = pyro.param("c_{}".format(card)).item()
    print(card)
    print(a, b, c, "*****" if card in game.players[game.turn].hand else "")
