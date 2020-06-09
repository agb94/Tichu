import torch
import torch.nn as nn
import torch.nn.functional as F

from tichu_env import *

'''Tichu-evaluating neural network, designed'''

class TichuNet1(nn.Module):
    '''unlike namesake doesn't include tichu encodings'''
    def __init__(self, card_emb_dim = 20, card_state_num = 7,
                 other_emb_dim = 5):
        super(TichuNet1, self).__init__()
        norm_card_type_num = len(Card.NUMBERS.keys())
        self.card_state_emb = nn.Embedding(card_state_num, card_emb_dim)
        self.owner_emb = nn.Embedding(NUM_PLAYERS, other_emb_dim)
        self.call_emb = nn.Embedding(norm_card_type_num+1, other_emb_dim)
        
        self.value_summarizer = nn.Sequential(
            nn.Conv2d(card_emb_dim, card_emb_dim, (4, 1)),
            nn.ReLU(),
            nn.Conv2d(card_emb_dim, card_emb_dim, 1),
            nn.ReLU()
        )
        self.straight_summarizer = nn.Sequential(
            nn.Conv2d(card_emb_dim, card_emb_dim, (1, 5)),
            nn.ReLU(),
            nn.Conv2d(card_emb_dim, card_emb_dim, 1),
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(13*card_emb_dim+2*other_emb_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.ced = card_emb_dim
        self.oed = other_emb_dim
    
    def forward(self, card_reps, noncard_reps):
        norm_cr, spec_cr = card_reps
        owner_idx, call_idx = noncard_reps
        norm_x = self.card_state_emb(norm_cr).permute(0, 3, 1, 2)
        spec_x = self.card_state_emb(spec_cr).permute(0, 3, 1, 2)

        norm_x = self.value_summarizer(norm_x)
        norm_x = self.straight_summarizer(norm_x)
        norm_x = norm_x.reshape(-1, 9*self.ced)
        spec_x = spec_x.reshape(-1, 4*self.ced)
        card_x = torch.cat([norm_x, spec_x], dim=1)

        owner_x = self.owner_emb(owner_idx).squeeze(1)
        call_x = self.call_emb(call_idx).squeeze(1)
        other_x = torch.cat([owner_x, call_x], dim=1)
        all_x = torch.cat([card_x, other_x], dim=1)

        pred_val = self.predictor(all_x)
        return pred_val

class TichuNet2(nn.Module):
    '''Incorporates cards left in each player's hand; better card representation'''
    def __init__(self, card_emb_dim = 20, card_state_num = 7,
                other_emb_dim = 5):
        super(TichuNet2, self).__init__()
        norm_card_type_num = len(Card.NUMBERS.keys())
        self.card_state_emb = nn.Embedding(card_state_num, card_emb_dim)
        self.owner_emb = nn.Embedding(NUM_PLAYERS, other_emb_dim)
        self.call_emb = nn.Embedding(norm_card_type_num+1, other_emb_dim)
        self.cards_left_emb = nn.Embedding(norm_card_type_num+1, other_emb_dim)
        
        self.value_summarizer = nn.Sequential(
            nn.Conv2d(card_emb_dim, card_emb_dim, (4, 1)),
            nn.ReLU(),
            nn.Conv2d(card_emb_dim, card_emb_dim, 1),
            nn.ReLU()
        )
        self.straight_summarizer = nn.Sequential(
            nn.Conv2d(card_emb_dim, card_emb_dim, (1, 5)),
            nn.ReLU(),
            nn.Conv2d(card_emb_dim, card_emb_dim, 1),
            nn.ReLU(),
            nn.Conv2d(card_emb_dim, card_emb_dim, (4, 1)),
            nn.ReLU(),
            nn.Conv2d(card_emb_dim, card_emb_dim, 1),
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(26*card_emb_dim+6*other_emb_dim, 100),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.ced = card_emb_dim
        self.oed = other_emb_dim

    def forward(self, card_reps, noncard_reps):
        norm_cr, spec_cr = card_reps
        owner_idx, call_idx, left_card_nums = noncard_reps
        norm_x = self.card_state_emb(norm_cr).permute(0, 3, 1, 2)
        spec_x = self.card_state_emb(spec_cr).permute(0, 3, 1, 2)

        val_x = self.value_summarizer(norm_x).reshape(-1, 13*self.ced)
        str_x = self.straight_summarizer(norm_x).reshape(-1, 9*self.ced)
        spec_x = spec_x.reshape(-1, 4*self.ced)
        card_x = torch.cat([val_x, str_x, spec_x], dim=1)

        owner_x = self.owner_emb(owner_idx).squeeze(1)
        call_x = self.call_emb(call_idx).squeeze(1)
        left_cards_x = self.cards_left_emb(left_card_nums).reshape(-1, 4*self.oed)
        other_x = torch.cat([owner_x, call_x, left_cards_x], dim=1)
        all_x = torch.cat([card_x, other_x], dim=1)

        pred_val = self.predictor(all_x)
        return pred_val

class TichuNet3b(nn.Module):
    '''Incorporates tichu, estimates which place player will be'''
    def __init__(self, card_emb_dim = 20, card_state_num = 8,
                other_emb_dim = 5):
        super(TichuNet3b, self).__init__()
        norm_card_type_num = len(Card.NUMBERS.keys())
        self.card_state_emb = nn.Embedding(card_state_num, card_emb_dim)
        self.owner_emb = nn.Embedding(NUM_PLAYERS, other_emb_dim)
        self.call_emb = nn.Embedding(norm_card_type_num+1, other_emb_dim)
        self.cards_left_emb = nn.Embedding(norm_card_type_num+1, other_emb_dim)
        self.tichu_emb = nn.Embedding(3, other_emb_dim)
        
        self.value_summarizer = nn.Sequential(
            nn.Conv2d(card_emb_dim, card_emb_dim, (4, 1)),
            nn.ReLU(),
            nn.Conv2d(card_emb_dim, card_emb_dim, 1),
            nn.ReLU()
        )
        self.straight_summarizer = nn.Sequential(
            nn.Conv2d(card_emb_dim, card_emb_dim, (1, 5)),
            nn.ReLU(),
            nn.Conv2d(card_emb_dim, card_emb_dim, 1),
            nn.ReLU(),
            nn.Conv2d(card_emb_dim, card_emb_dim, (4, 1)),
            nn.ReLU(),
            nn.Conv2d(card_emb_dim, card_emb_dim, 1),
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(26*card_emb_dim+10*other_emb_dim, 200),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(100, 1+4)
        )
        self.ced = card_emb_dim
        self.oed = other_emb_dim

    def forward(self, card_reps, noncard_reps, ret_form='all'):
        norm_cr, spec_cr = card_reps
        owner_idx, call_idx, left_card_nums, tichu_states = noncard_reps
        norm_x = self.card_state_emb(norm_cr).permute(0, 3, 1, 2)
        spec_x = self.card_state_emb(spec_cr).permute(0, 3, 1, 2)

        val_x = self.value_summarizer(norm_x).reshape(-1, 13*self.ced)
        str_x = self.straight_summarizer(norm_x).reshape(-1, 9*self.ced)
        spec_x = spec_x.reshape(-1, 4*self.ced)
        card_x = torch.cat([val_x, str_x, spec_x], dim=1)

        owner_x = self.owner_emb(owner_idx).squeeze(1)
        call_x = self.call_emb(call_idx).squeeze(1)
        left_cards_x = self.cards_left_emb(left_card_nums).reshape(-1, 4*self.oed)
        tichu_x = self.cards_left_emb(tichu_states).reshape(-1, 4*self.oed)
        other_x = torch.cat([owner_x, call_x, left_cards_x, tichu_x], dim=1)
        all_x = torch.cat([card_x, other_x], dim=1)

        x = self.predictor(all_x)
        pred_val = x[:, :1]
        pred_place_logits = x[:, 1:]
        if ret_form == 'value':
            return pred_val
        elif ret_form == 'place_odds':
            return nn.Softmax(dim=1)(pred_place_logits)
        elif ret_form == 'all':
            return pred_val, pred_place_logits
        else:
            raise ValueError(f'Unknown return form {ret_form}')