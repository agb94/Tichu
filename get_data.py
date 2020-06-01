from tichu_env import Game
from model import TichuNet1
from ai import NeuralPlayer, RandomPlayer

import torch
import pickle as pkl
import tqdm

def recursive_numpy(obj):
    if type(obj) == tuple:
        return tuple(recursive_numpy(o) for o in obj)
    else:
        return obj.numpy()

def neural_player_generator(x, y):
    return NeuralPlayer(
        x, y, net, 
        recording=True, 
        default_temp = agent_temperature, 
        device=device
    )

def target_calculator(rotated_final, scores):
    adjusted_score = [ts - ps for ts, (_, ps) in zip(rotated_final, scores)]
    our_score = adjusted_score[0] + adjusted_score[2]
    their_score = adjusted_score[1] + adjusted_score[3]
    return our_score - their_score

# parameters
device = 'cuda'
game_num = 1000
save_dir = './data/'
save_filename = 'phase2.pkl'
net_path = './models/phase1_net.pth'
agent_temperature = 3.

net = TichuNet1()
net.load_state_dict(torch.load(net_path))
net.to(device)

input_data = []
target_data = []
for game_num in tqdm.trange(game_num):
    robot_players = [neural_player_generator for _ in range(3)] + [RandomPlayer]
    game = Game(players = robot_players)
    final_score = game.run_game()

    for player in game.players[:3]:
        rotated_final = final_score[player.player_id:] + final_score[:player.player_id]
        for state_rep, scores in player.export_data():
            input_data.append(recursive_numpy(state_rep))
            target_value = target_calculator(rotated_final, scores)
            target_data.append(target_value)

save_dict = {
    'input': input_data,
    'target': target_data
}

pkl.dump(save_dict, open(save_dir+save_filename, 'wb'))
