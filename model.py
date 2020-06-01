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
        