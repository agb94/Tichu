import os
import argparse
import torch
import numpy as np
import random
import json
import operator
from tqdm import tqdm
from copy import deepcopy
from model import TichuNet2, TichuNet3b
from tichu_env import *
from ai import RandomPlayer, GreedyPlayer, PatientGreedyPlayer, NeuralPlayer
from UI import *
from scipy.stats import rankdata
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from inference3 import *

def main(args):
    print(args)
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
        tn1.eval()
        tmp_player = lambda g, i: NeuralPlayer(g, i, device=args.device, default_temp=.1)
        players = [
            lambda x, y: NeuralPlayer(x, y, tn1, device=args.device, default_temp=.1)
            for _ in range(NUM_PLAYERS)
        ]

    # Initialize Game
    game = Game(args.gameseed, players)

    game.players[0].add_cards_to_hand([
        Card('MahJong'), Card('Red', 3), Card('Blue', 3), Card('Blue', 6),
        Card('Green', 7), Card('Red', 8), Card('Blue', 8), Card('Green', 9),
        Card('Red', 9), Card('Green', 10), Card('Blue', 'J'), Card('Blue', 'K'),
        Card('Green', 'A'), Card('Dragon')])

    game.players[1].add_cards_to_hand([
        Card('Black', 2), Card('Green', 3), Card('Black', 3), Card('Blue', 4),
        Card('Green', 4), Card('Red', 4), Card('Blue', 5), Card('Blue', 7),
        Card('Green', 8), Card('Blue', 9), Card('Red', 'J'), Card('Green', 'Q'),
        Card('Black', 'Q'), Card('Red', 'A')])

    game.players[2].add_cards_to_hand([
        Card('Blue', 2), Card('Green', 2), Card('Green', 5), Card('Black', 5),
        Card('Red', 6), Card('Green', 6), Card('Black', 7), Card('Red', 7),
        Card('Black', 'J'), Card('Green', 'J'), Card('Red', 'Q'), Card('Green', 'K'),
        Card('Black', 'A'), Card('Phoenix')])

    game.players[3].add_cards_to_hand([
        Card('Dog'), Card('Red', 2), Card('Black', 4), Card('Red', 5),
        Card('Black', 6), Card('Black', 8), Card('Black', 9), Card('Red', 10),
        Card('Black', 10), Card('Blue', 10), Card('Blue', 'Q'), Card('Black', 'K'),
        Card('Red', 'K'), Card('Blue', 'A')])
    assert len(set(sum([p.hand for p in game.players], []))) == 56

    game.mark_exchange(0, 1, 2)
    game.mark_exchange(0, 2, 11)
    game.mark_exchange(0, 3, 1)

    game.mark_exchange(1, 0, 3)
    game.mark_exchange(1, 2, 0)
    game.mark_exchange(1, 3, 10)

    game.mark_exchange(2, 0, 10)
    game.mark_exchange(2, 1, 0)
    game.mark_exchange(2, 3, 1)

    game.mark_exchange(3, 0, 1)
    game.mark_exchange(3, 1, 10)
    game.mark_exchange(3, 2, 0)

    game.exchange()

    initial_state = deepcopy(game)
    no_call_cards = []

    print(initial_state)

    histories = [
        (0, MahJongSingle(Card('MahJong'), 14)),
        (1, Single(Card('Red', 'A'))),
        (2, None),
        (3, None),
        (0, Single(Card('Dragon'))),
        (1, None),
        (2, None),
        (3, None),
        (0, Straight(
                Card('Blue', 6), Card('Green', 7), Card('Red', 8), Card('Green', 9),
                Card('Green', 10), Card('Blue', 'J'), Card('Red', 'Q'))),
        (1, None),
        (2, None),
        (3, None),
        (0, Single(Card('Red', 2))),
        (1, Single(Card('Blue', 5))),
        (2, None),
        (3, Single(Card('Black', 8))),
        (0, Single(Card('Red', 9))),
        (1, None),
        (2, None),
        (3, Single(Card('Red', 'J'))),
        (0, Single(Card('Green', 'A'))),
        (1, None),
        (2, None),
        (3, None),
        (0, Single(Card('Blue', 4))),
        (1, Single(Card('Blue', 7))),
        (2, None),
        (3, Single(Card('Black', 9))),
        (0, None),
        (1, None),
        (2, None),
        (3, Straight(
                Card('Green', 2), Card('Red', 3), Card('Black', 4),
                Card('Red', 5), Card('Black', 6))),
        (0, None),
        (1, None),
        (2, None),
        (3, FullHouse(
                Pair(Card('Black', 'K'), Card('Red', 'K')),
                Triple(Card('Red', 10), Card('Black', 10), Card('Blue', 10)))),
        (0, None),
        (1, None),
        (2, None),
        (3, Single(Card('Blue', 'A'))),
        (0, None),
        (1, None),
        (2, None),
        (3, None),
        (0, Single(Card('Blue', 8)))
    ]
    # Play Game
    for t in range(args.run):
        action_probs = game.players[game.turn].action_probs()
        if t < len(histories):
            real_action = histories[t][1]
        else:
            real_action = game.players[game.turn].sample_action()
        if not game.call_satisifed and game.play_logs:
            if not game.current or isinstance(game.current[-1], MahJongSingle) or (isinstance(game.current[-1], Single) and game.current[-1].value < game.call_value):
                if real_action is None or not any([c.value == game.call_value for c in real_action.cards]):
                    no_call_cards.append(game.turn)
                    print("[game log] It is clear that {} does not have call cards".format(game.turn))
        current_top = game.current[-1] if game.current else None
        print("[game log] current top: {}\n              player {}: {}".format(current_top, game.turn, coloring(real_action, bcolors.OKGREEN)))
        for a, p in reversed(sorted(action_probs, key=operator.itemgetter(1))):
            print("\t\t\t-", str(a) if a != real_action else coloring(a, bcolors.OKGREEN), round(float(p), 7))
        game.play(game.turn, real_action)

    print()

    print("Start sampling...")
    # Inference
    unknown_cards = list(set(game.unused_cards) - set(game.players[args.observer].card_locs.keys()))
    unknown_cards.sort()
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
        actual.append([float(pid == holder[card]) for pid in range(NUM_PLAYERS)])
    predicted, actual, uniform = np.array(predicted), np.array(actual), np.array(uniform)
    print(predicted)
    print(uniform)
    print(actual)

    metrics = { "AP": average_precision_score, "ROC AUC": roc_auc_score }

    rand = np.random.rand(actual.shape[0], actual.shape[1])
    rand[:, 0] = .0
    for i in range(actual.shape[0]):
        rand[i] = rand[i]/np.sum(rand[i])
    print(rand)
    """
    Per player
    """
    print("Per player")
    per_player_total = {metric: {"IS": [], "Uniform": [], "Random": []} for metric in metrics}
    for pid in range(NUM_PLAYERS):
        if pid == args.observer or hand_size[pid] == 0:
            continue
        print("player {}".format(pid))
        prediction_scores = {
            "IS": predicted[:, pid],
            "Uniform": uniform[:, pid],
            "Random": rand[:, pid]
        }
        for metric in metrics:
            for ptype in prediction_scores:
                score = metrics[metric](actual[:, pid], prediction_scores[ptype])
                print(f"{ptype:10} {metric:10}: {score}")
                per_player_total[metric][ptype].append(score)
        print("==================================================================")
    # Aggregation
    per_player_aggr = {metric: {"IS": None, "Uniform": None, "Random": None} for metric in metrics}
    for metric in per_player_total:
        for ptype in per_player_total[metric]:
            averaged = np.mean(per_player_total[metric][ptype])
            per_player_aggr[metric][ptype] = float(averaged)
            s = f"{ptype:10} {metric:10}: {averaged}"
            if ptype == "IS":
                s = coloring(s, bcolors.WARNING)
            print(s)
    print("==================================================================")

    """
    Per card
    """
    print("Per card")
    per_card_total = {metric: {"IS": [], "Uniform": [], "Random": []} for metric in metrics}
    for i in range(actual.shape[0]):
        #print(f"Card {unknown_cards[i]}")
        prediction_scores = {
            "IS": predicted[i, :],
            "Uniform": uniform[i, :],
            "Random": rand[i, :]
        }
        for metric in metrics:
            for ptype in prediction_scores:
                score = metrics[metric](actual[i, :], prediction_scores[ptype])
                #print(f"{ptype:10} {metric:10}: {score}")
                per_card_total[metric][ptype].append(score)
        #print("==================================================================")

    # Aggregation
    per_card_aggr = {metric: {"IS": None, "Uniform": None, "Random": None} for metric in metrics}
    for metric in per_card_total:
        for ptype in per_card_total[metric]:
            averaged = np.mean(per_card_total[metric][ptype])
            per_card_aggr[metric][ptype] = float(averaged)
            s = f"{ptype:10} {metric:10}: {averaged}"
            if ptype == "IS":
                s = coloring(s, bcolors.WARNING)
            print(s)
    print("==================================================================")

    """
    All together
    """
    prediction_scores = {
        "IS": predicted.reshape(1, -1)[0],
        "Uniform": uniform.reshape(1, -1)[0],
        "Random": rand.reshape(1, -1)[0]
    }
    all_aggr = {metric: {"IS": None, "Uniform": None, "Random": None} for metric in metrics}
    for metric in metrics:
        for ptype in prediction_scores:
            score = metrics[metric](actual.reshape(1, -1)[0], prediction_scores[ptype])
            print(f"{ptype:10} {metric:10}: {score}")
            all_aggr[metric][ptype] = float(score)
    print("==================================================================")

    sample_data = {
        "args": vars(args),
        "results": {
            "per_player": per_player_aggr,
            "per_card":   per_card_aggr,
            "all":        all_aggr
        }
    }
    try:
        with open(f"human_samples/{args.player}_{args.samples}_{args.run}_{args.observer}_{args.id}.json", "w") as json_file:
            json.dump(sample_data, json_file)
    except:
        os.remove(f"human_samples/{args.player}_{args.samples}_{args.run}_{args.observer}_{args.id}.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default="phase3_net3b.pth")
    parser.add_argument('--device', '-d', type=str, default="cpu")
    parser.add_argument('--player', '-p', type=str, default="greedy")
    parser.add_argument('--gameseed', '-g', type=int, default=None)
    parser.add_argument('--observer', '-o', type=int, default=0)
    parser.add_argument('--run', '-r', type=int, default=10)
    parser.add_argument('--samples', '-s', type=int, default=1000)
    parser.add_argument('--id', '-i', type=int, default=0)

    args = parser.parse_args()
    main(args)

#main(Namespace({'model': None, 'observer': 0, 'player': 'random'}))
