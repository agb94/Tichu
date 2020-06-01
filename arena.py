'''Fight!'''

import tqdm
import torch

from tichu_env import Game
from model import TichuNet1
from ai import RandomPlayer, NeuralPlayer, HumanPlayer

def neural_player_generator(x, y, net):
    return NeuralPlayer(
        x, y, net, 
        recording=True, 
        default_temp = 5, 
        device=device
    )

game_number = 200
device = 'cuda'
team1_label = 'Random'
team2_label = 'phase2'
team1_net_path = './models/phase1_net.pth'
team2_net_path = './models/phase2_net.pth'
verbose = False

# Team1Player = RandomPlayer
team1_net = TichuNet1()
team1_net.load_state_dict(torch.load(team1_net_path))
team1_net.to(device)
Team1Player = lambda x, y: neural_player_generator(x, y, team1_net)

team2_net = TichuNet1()
team2_net.load_state_dict(torch.load(team2_net_path))
team2_net.to(device)
Team2Player = lambda x, y: neural_player_generator(x, y, team2_net)
# Team2Player = RandomPlayer

team1_score = 0
team2_score = 0
team1_wins = 0
team2_wins = 0

for game_num in tqdm.trange(game_number):
    robot_players = [Team1Player, Team2Player, Team1Player, Team2Player]
    game = Game(players = robot_players)
    
    final_score = game.run_game(verbose=verbose)
    game_team1_score = final_score[0]+final_score[2]
    game_team2_score = final_score[1]+final_score[3]
    team1_score += game_team1_score
    team2_score += game_team2_score
    
    if game_team1_score > game_team2_score:
        team1_wins += 1
    else:
        team2_wins += 1

print(f'Team 1 ({team1_label}) won {team1_wins}/{game_number}, total {team1_score} points')
print(f'Team 2 ({team2_label}) won {team2_wins}/{game_number}, total {team2_score} points')