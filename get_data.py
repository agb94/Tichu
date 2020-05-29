from tichu_env import Game
from model import TichuNet1
from ai import NeuralPlayer

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

# parameters
device = 'cuda'
game_num = 500
save_dir = './data/'
save_filename = 'phase1.pkl'
agent_temperature = 10.

net = TichuNet1()
net.to(device)

input_data = []
target_data = []
for game_num in tqdm.trange(game_num):
    robot_players = [neural_player_generator for _ in range(4)]
    game = Game(players = robot_players)
    final_score = game.run_game()

    for player in game.players:
        rotated_final = final_score[player.player_id:] + final_score[:player.player_id]
        for state_rep, scores in player.export_data():
            input_data.append(recursive_numpy(state_rep))
            adjusted_score = [ts - ps for ts, (_, ps) in zip(rotated_final, scores)]
            target_data.append(adjusted_score)

save_dict = {
    'input': input_data,
    'target': target_data
}

pkl.dump(save_dict, open(save_dir+save_filename, 'wb'))
