import random
import numpy as np

from tichu_env import *
import torch

class AutonomousPlayer(Player):
    def action_probs(self):
        '''Returns list of actions and their odds.'''
        raise NotImplementedError

    def sample_action(self):
        '''Samples action based on odds calculated from action_probs'''
        actions, odds = zip(*self.action_probs())
        rand_idx = np.random.choice(len(actions), 1, p=odds)[0]
        return actions[rand_idx]

    def choose_exchange(self):
        '''Provides list of (user index, card index to send)'''
        raise NotImplementedError
    
    def call_big_tichu(self):
        raise NotImplementedError

class RandomPlayer(AutonomousPlayer):
    '''Simple player that performs random actions.'''

    def action_probs(self):
        my_options = self.possible_actions()
        action_num = len(my_options)
        any_option_odds = 1./action_num
        return [(action, any_option_odds) for action in my_options]

    def choose_exchange(self):
        other_idxs = list(filter(lambda x: x != self.player_id, range(4)))
        picked_cards = random.sample(range(len(self.hand)), 3)
        return list(zip(other_idxs, picked_cards))
    
    def call_big_tichu(self):
        baseline_odds = 0.1
        if random.random() < baseline_odds:
            return True
        else:
            return False

class NeuralPlayer(AutonomousPlayer):
    '''Player that relies on value network to compute next move'''
    def __init__(self, game, player_id, network, 
                 default_temp = 1.0, device='cuda',
                 recording=False, debug=False):
        super().__init__(game, player_id)
        self.network = network
        self.device = device
        self.recording = recording
        self.debug = debug
        self.default_temp = default_temp
        if recording:
            self.records = []
    
    def normalized_playerid(self, id):
        return (id - self.player_id) % NUM_PLAYERS

    def card_rep(self, action):
        '''Returns game state in a tensor format.'''
        # card information
        norm_card_states = np.zeros((4, 13))
        spec_card_states = np.zeros((4, 1))
        
        modified_hand = set(self.hand)
        curr_cards = set(sum([c.cards for c in self.game.current], []))
        if action is not None:
            modified_hand -= set(action.cards)
            curr_cards |= set(action.cards)
            
        used_cards = set(self.game.used)
        known_cards = set(self.card_locs.keys())
        # card state key: 0 in my hand; 1 in current play; 2 played; 
        # 3-5 known but unused; 6 unknown
        for card in self.game.deck.cards:
            if card in modified_hand: # not sure if ok
                state_val = 0
            elif card in curr_cards:
                state_val = 1
            elif card in used_cards:
                state_val = 2
            elif card in known_cards:
                card_owner = self.card_locs[card]
                state_val = self.normalized_playerid(card_owner) + 2
            elif card in self.game.unused_cards:
                state_val = 6
            else:
                raise ValueError(f'{card} is of unknown state')
            
            if card.number is None:
                suite_idx = Card.SPECIALS.index(card.suite)
                spec_card_states[suite_idx, 0] = state_val
            else:
                suite_idx = Card.COLORS.index(card.suite)
                number_idx = Card.NUMBERS[card.number] - 2
                norm_card_states[suite_idx, number_idx] = state_val
        
        return norm_card_states, spec_card_states
    
    def noncard_rep(self, action):
        if action is None:
            curr_owner = (self.game.turn - (self.game.pass_count+1)) % 4
            owner_rep = self.normalized_playerid(curr_owner)
            call_value = 0
        else:
            owner_rep = 0 # if not skip, any action incurs my owning pile (I think).
            if isinstance(action, MahJongSingle):
                call_value = action.call_value - 1
            else:
                call_value = 0
        return owner_rep, call_value

    @classmethod
    def state2Tensor(cls, card_rep, noncard_rep, device='cuda'):
        norm_cr, spec_cr = card_rep
        owner_idx, call_idx = noncard_rep
        norm_cr = torch.LongTensor(norm_cr).unsqueeze(0).to(device)
        spec_cr = torch.LongTensor(spec_cr).unsqueeze(0).to(device)

        owner_idx = torch.LongTensor([owner_idx]).unsqueeze(0).to(device)
        call_idx = torch.LongTensor([call_idx]).unsqueeze(0).to(device)

        return (norm_cr, spec_cr), (owner_idx, call_idx)

    def action_value(self, action):
        action_card_rep = self.card_rep(action)
        action_noncard_rep = self.noncard_rep(action)
        tensor_cr, tensor_ncr = self.__class__.state2Tensor(
            action_card_rep, action_noncard_rep, device=self.device
        )
        estimated_value = self.network(tensor_cr, tensor_ncr).item()
        return estimated_value
    
    def action_probs(self, softmax_T = None):
        my_options = self.possible_actions()
        action_values = np.array([self.action_value(action) for action in my_options])
        action_logits = action_values - np.max(action_values) # numeric stability
        if softmax_T is None:
            action_probs = np.exp(action_logits/self.default_temp)
        else:
            action_probs = np.exp(action_logits/softmax_T)
        action_probs = action_probs/np.sum(action_probs)
        assert (np.sum(action_probs) - 1) < 1e-5
        
        if self.debug:
            print('----')
            for a_idx, action in enumerate(my_options):
                print(f'{str(action)} | estimated gain {action_values[a_idx]:.3f}, probability {action_probs[a_idx]:.3f}')
            print('----')
        return [(my_options[i], action_probs[i]) for i in range(len(my_options))]
    
    def sample_action(self):
        # reimplemented for recording
        sampled_action = super().sample_action()
        if self.recording:
            self.record(sampled_action)
        return sampled_action
                
    def call_big_tichu(self):
        baseline_odds = 0.0
        if random.random() < baseline_odds:
            return True
        else:
            return False
    
    def choose_exchange(self):
        other_idxs = list(filter(lambda x: x != self.player_id, range(4)))
        picked_cards = random.sample(range(len(self.hand)), 3)
        return list(zip(other_idxs, picked_cards))
    
    def record(self, action):
        action_card_rep = self.card_rep(action)
        action_noncard_rep = self.noncard_rep(action)
        state_reps = self.__class__.state2Tensor(
            action_card_rep, action_noncard_rep, device='cpu'
        )

        scores = []
        for player in self.game.players:
            norm_pid = self.normalized_playerid(player.player_id)
            scores.append((norm_pid, Game.card_scorer(player.obtained)))
        self.records.append((state_reps, scores))
    
    def export_data(self):
        return self.records

class HumanPlayer(Player):
    def print_hand(self):
        self.hand.sort()
        print('Your cards are:')
        for c_idx, card in enumerate(self.hand):
            print(c_idx, card)

    def sample_action(self):
        self.print_hand()
        actions = self.possible_actions()
        print('You have the following available actions:')
        for a_idx, action in enumerate(actions):
            print(f' {a_idx} | {str(action)}')
        idx = self._secure_input('Which one will you play? ', int)
        return actions[idx]
    
    def call_big_tichu(self):
        self.print_hand()
        user_decision = input('Would you like to play big tichu (y/n)? ')
        if 'y' in user_decision:
            return True
        else:
            return False
    
    def _secure_input(self, query_str, f):
        while True:
            try:
                return f(input(query_str))
            except ValueError:
                print('Unknown input format, please try again.')

    def choose_exchange(self):
        self.print_hand()
        print(f'You are player #{self.player_id}')
        other_idxs = list(filter(lambda x: x != self.player_id, range(4)))
        exchange_list = []
        for idx in other_idxs:
            card_id = self._secure_input(f'Choose card to send to player {idx}: ', int)
            exchange_list.append((idx, card_id))
        return exchange_list

def test_player():
    from tichu_env import Game
    from model import TichuNet1
    device = 'cuda'
    tn1 = TichuNet1()
    tn1.to(device)
    robot_players = [lambda x, y: NeuralPlayer(x, y, tn1, device=device) 
                     for _ in range(3)]
    all_players = robot_players + [HumanPlayer]
    random.shuffle(all_players)
    game = Game(0, all_players)
    game.run_game(upto='scoring', verbose=True)
    print(game)

if __name__ == '__main__':
    test_player()
