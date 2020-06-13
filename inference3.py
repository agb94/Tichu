import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy
from model import TichuNet2, TichuNet3b
from tichu_env import *
from ai import RandomPlayer, GreedyPlayer, NeuralPlayer
from UI import *

HAND_SIZE = len(Deck().cards) / NUM_PLAYERS

def get_observed_hands(game, observer):
    # Return cards of which initial locations are already observed by observer
    hands = [list() for pid in range(NUM_PLAYERS)]
    card_locs = game.players[observer].card_locs
    for card in card_locs:
        hands[card_locs[card]].append(card)
    return hands

def sample_and_weight(game, observer, initial_state, no_call_cards):
    observed_hands = get_observed_hands(game, args.observer)
    # Sampling
    hands = deepcopy(observed_hands)
    if no_call_cards:
        for card in [card for card in game.deck.cards if card.value == game.call_value and card not in sum(hands, [])]:
            assigned = False
            while not assigned:
                player_id = random.choice(range(0, NUM_PLAYERS))
                if player_id not in no_call_cards and len(hands[player_id]) < HAND_SIZE:
                    hands[player_id].append(card)
                    assigned = True

    for card in [card for card in game.deck.cards if card not in sum(hands, [])]:
        assigned = False
        while not assigned:
            player_id = random.choice(range(0, NUM_PLAYERS))
            if len(hands[player_id]) < HAND_SIZE:
                hands[player_id].append(card)
                assigned = True

    for i, hand in enumerate(hands):
        unused = list(filter(lambda c: c not in initial_state.used, hand))
        initial_state.players[i].reset_hand(unused)

    replay_start_point = len(initial_state.play_logs)
    play_logs = game.play_logs[replay_start_point:]
    if not play_logs:
        return None

    assert initial_state.turn == play_logs[0][0]

    probs = []
    for player_id, combi, call_satisifed in play_logs:
        action_probs = initial_state.players[player_id].action_probs()
        prob = [item[1] for item in action_probs if item[0] == combi]
        if len(prob) != 1:
            return None
        probs.append(prob[0])
        initial_state.play(player_id, combi)

    return hands, float(np.prod(probs))

def main(args):
    my_id = args.observer
    if args.player == 'random':
        tmp_player = lambda g, i: RandomPlayer(g, i)
        players = [RandomPlayer for _ in range(NUM_PLAYERS)]
    if args.player == 'greedy':
        tmp_player = lambda g, i: GreedyPlayer(g, i)
        players = [GreedyPlayer for _ in range(NUM_PLAYERS)]
    elif args.player == 'neural':
        tn1 = TichuNet3b()
        if args.model:
            tn1.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        tmp_player = lambda g, i: NeuralPlayer(g, i, device=device, default_temp=.1)
        players = [
            lambda x, y: NeuralPlayer(x, y, tn1, device=device, default_temp=.1)
            for _ in range(NUM_PLAYERS)
        ]

    # Initialize Game
    game = Game(args.gameseed, players)
    game.run_game(upto='firstAction')
    initial_state = deepcopy(game)
    no_call_cards = []

    print(initial_state)

    # Play Game
    for t in range(args.run):
        real_action = game.players[game.turn].sample_action()
        if not game.call_satisifed:
            if not game.current or isinstance(game.current[-1], MahJongSingle) or (isinstance(game.current[-1], Single) and game.current[-1].value < game.call_value):
                if not any([c.value == game.call_value for c in real_action.cards]):
                    no_call_cards.append(game.turn)
                    print("[game log] It is clear that {} does not have call cards".format(game.turn))
        current_top = game.current[-1] if game.current else None
        print("[game log] current top: {}, player {}: {}".format(current_top, game.turn, real_action))
        game.play(game.turn, real_action)

    print()

    print("Start sampling...")
    # Inference
    unknown_cards = set(game.unused_cards) - set(game.players[args.observer].card_locs.keys())
    cards_weights = {c: [list() for _ in range(NUM_PLAYERS)] for c in unknown_cards}
    all_weights = []

    num_samples = args.samples
    for step in tqdm(range(num_samples)):
        result = sample_and_weight(game, args.observer, deepcopy(initial_state), no_call_cards)
        if result is None:
            continue
        sample, weight = result
        all_weights.append(weight)
        for card in cards_weights:
            for player_id in range(NUM_PLAYERS):
                if card in sample[player_id]:
                    cards_weights[card][player_id].append(weight)
                else:
                    cards_weights[card][player_id].append(.0)
    holder = {}
    for player_id in range(NUM_PLAYERS):
        for card in initial_state.players[player_id].hand:
            holder[card] = player_id

    weight_sum = np.sum(all_weights)
    for card in sorted(cards_weights.keys()):
        weights = np.array(cards_weights[card])
        weights = np.round(np.sum(weights, axis=1)/weight_sum, 4)
        #print(card, weights, holder[card])
        print(str(card) + "\t" + "\t".join([(str(weights[pid]) if pid != holder[card] else coloring(str(weights[pid]), bcolors.OKBLUE)) for pid in range(NUM_PLAYERS)]))
#    print(cards_weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default=None)
    parser.add_argument('--device', '-d', type=str, default="cpu")
    parser.add_argument('--player', '-p', type=str, default="greedy")
    parser.add_argument('--gameseed', '-g', type=int, default=0)
    parser.add_argument('--observer', '-o', type=int, default=0)
    parser.add_argument('--run', '-r', type=int, default=10)
    parser.add_argument('--samples', '-s', type=int, default=1000)
    args = parser.parse_args()
    main(args)

#main(Namespace({'model': None, 'observer': 0, 'player': 'random'}))
