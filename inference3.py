import argparse
import torch
import numpy as np
import random
import operator
from tqdm import tqdm
from copy import deepcopy
from model import TichuNet2, TichuNet3b
from tichu_env import *
from ai import RandomPlayer, GreedyPlayer, PatientGreedyPlayer, NeuralPlayer
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
                pid = random.choice(range(0, NUM_PLAYERS))
                if pid not in no_call_cards and len(hands[pid]) < HAND_SIZE:
                    hands[pid].append(card)
                    assigned = True

    for card in [card for card in game.deck.cards if card not in sum(hands, [])]:
        assigned = False
        while not assigned:
            pid = random.choice(range(0, NUM_PLAYERS))
            if len(hands[pid]) < HAND_SIZE:
                hands[pid].append(card)
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
    for pid, combi, call_satisifed in play_logs:
        if pid != observer:
            action_probs = initial_state.players[pid].action_probs()
            assert (np.sum([p for a, p in action_probs]) - 1) < 1e-5
            prob = [item[1] for item in action_probs if item[0] == combi]
            if len(prob) != 1:
                return None
            probs.append(prob[0])
        initial_state.play(pid, combi)

    return hands, float(np.sum(np.log(probs)))

def main(args):
    my_id = args.observer
    if args.player == 'random':
        tmp_player = lambda g, i: RandomPlayer(g, i)
        players = [RandomPlayer for _ in range(NUM_PLAYERS)]
    elif args.player == 'greedy':
        tmp_player = lambda g, i: GreedyPlayer(g, i)
        players = [GreedyPlayer for _ in range(NUM_PLAYERS)]
    elif args.player == 'patientgreedy':
        tmp_player = lambda g, i: PatientGreedyPlayer(g, i)
        players = [PatientGreedyPlayer for _ in range(NUM_PLAYERS)]
    elif args.player == 'neural':
        tn1 = TichuNet3b()
        if args.model:
            tn1.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        tmp_player = lambda g, i: NeuralPlayer(g, i, device=args.device, default_temp=.1)
        players = [
            lambda x, y: NeuralPlayer(x, y, tn1, device=args.device, default_temp=.1)
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
        #action_probs = game.players[game.turn].action_probs()
        #real_action = sorted(action_probs, key=operator.itemgetter(1))[-1][0]
        real_action = game.players[game.turn].sample_action()
        if not game.call_satisifed:
            if not game.current or isinstance(game.current[-1], MahJongSingle) or (isinstance(game.current[-1], Single) and game.current[-1].value < game.call_value):
                if real_action is None or not any([c.value == game.call_value for c in real_action.cards]):
                    no_call_cards.append(game.turn)
                    print("[game log] It is clear that {} does not have call cards".format(game.turn))
        current_top = game.current[-1] if game.current else None
        print("[game log] current top: {}\n              player {}: {}".format(current_top, game.turn, coloring(real_action, bcolors.OKGREEN)))
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
        sample, log_weight = result
        weight = np.e**log_weight
        all_weights.append(weight)
        for card in cards_weights:
            for pid in range(NUM_PLAYERS):
                if card in sample[pid]:
                    cards_weights[card][pid].append(weight)
                else:
                    cards_weights[card][pid].append(.0)
    holder = {}
    for pid in range(NUM_PLAYERS):
        for card in initial_state.players[pid].hand:
            holder[card] = pid

    hand_size = [len([card for card in game.players[pid].hand if card in unknown_cards]) for pid in range(NUM_PLAYERS)]
    hand_ratio = [s/sum(hand_size) for s in hand_size]
    print(hand_ratio)
    weight_sum = np.sum(all_weights)
    predicted, actual, uniform = [], [], []
    for card in sorted(cards_weights.keys()):
        weights = np.array(cards_weights[card])
        weights = np.round(np.sum(weights, axis=1)/weight_sum, 4)
        print(str(card) + "\t" + "\t".join([(str(weights[pid]) if pid != holder[card] else coloring(str(weights[pid]), bcolors.OKBLUE)) for pid in range(NUM_PLAYERS)]))
        uniform.append(hand_ratio)
        predicted.append(weights)
        actual.append([int(pid == holder[card]) for pid in range(NUM_PLAYERS)])
    predicted, actual, uniform = np.array(predicted), np.array(actual), np.array(uniform)
    print(predicted)
    print(uniform)
    print(actual)
    for pid in range(NUM_PLAYERS):
        print("player {}".format(pid))
        print("     IS Brier score:", np.sum(np.power(predicted[:, pid] - actual[:, pid], 2))/actual.shape[0])
        print("   Size Brier score:", np.sum(np.power(uniform[:, pid] - actual[:, pid], 2))/actual.shape[0])
    print("total")
    print("     IS Brier score:", np.sum(np.power(predicted[:] - actual[:], 2))/actual.shape[0])
    print("   Size Brier score:", np.sum(np.power(uniform[:] - actual[:], 2))/actual.shape[0])

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
