import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

from tichu_env import *

def model(game, observer, action):
    if game.turn == observer:
        return None
    known_cards = set(game.players[observer].card_locs.keys())
    unknown_cards = set(game.unused_cards) - known_cards
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
        a = torch.tensor(10.0)
        b = torch.tensor(10.0)
        c = torch.tensor(10.0)
        player_probs = pyro.sample('{}_probs'.format(card), dist.Dirichlet(torch.stack([a, b, c])))
        normalized_player_probs = player_probs #/ torch.sum(player_probs)
        print(card, normalized_player_probs)
        probs.append(normalized_player_probs)
    probs = torch.stack(probs)

    hands = [list() for p in other_players]
    card_probs = [list() for p in other_players]
    for i, card in random.sample(tuple(enumerate(unknown_cards)), len(unknown_cards)):
        assigned = False
        while not assigned:
            player = pyro.sample('card_{}'.format(card), dist.Categorical(probs=probs[i]))
            if len(hands[int(player)]) < num_hand_cards[int(player)]:
                hands[int(player)].append(card)
                card_probs[int(player)].append(probs[i][int(player)])
                assigned = True

    for i, card in enumerate(hands[player_idx]):
        print(card, card_probs[player_idx][i])
        pass

    possible_actions = Player._get_possible_actions(game, hands[player_idx], None)
    action_probs = torch.ones(len(possible_actions)) / len(possible_actions)
    if action in possible_actions:
        prob = action_probs[possible_actions.index(action)]
    else:
        prob = .0
    return prob

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
print(model(game, my_id, game.players[game.turn].possible_actions()[0]))
print(model(game, my_id, game.players[game.turn].possible_actions()[0]))
print(model(game, my_id, game.players[game.turn].possible_actions()[0]))
print(model(game, my_id, game.players[game.turn].possible_actions()[0]))
print(model(game, my_id, game.players[game.turn].possible_actions()[0]))
print(model(game, my_id, game.players[game.turn].possible_actions()[0]))
print(model(game, my_id, game.players[game.turn].possible_actions()[0]))
print(model(game, my_id, game.players[game.turn].possible_actions()[0]))
print(model(game, my_id, game.players[game.turn].possible_actions()[0]))
print(model(game, my_id, game.players[game.turn].possible_actions()[0]))
print(model(game, my_id, game.players[game.turn].possible_actions()[0]))
print(model(game, my_id, game.players[game.turn].possible_actions()[0]))
print(model(game, my_id, game.players[game.turn].possible_actions()[0]))
print(model(game, my_id, game.players[game.turn].possible_actions()[0]))
print(model(game, my_id, game.players[game.turn].possible_actions()[0]))
print(model(game, my_id, game.players[game.turn].possible_actions()[0]))
print(model(game, my_id, game.players[game.turn].possible_actions()[0]))
print(model(game, my_id, game.players[game.turn].possible_actions()[0]))
