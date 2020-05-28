import random
import numpy as np
from collections import defaultdict
from itertools import combinations
from abc import *

INF = 1000
NUM_PLAYERS = 4

class Card:
    SPECIALS = ["MahJong", "Phoenix", "Dragon", "Dog"]
    COLORS = ["Black", "Red", "Green", "Blue"]
    NUMBERS = {
        2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
        10: 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }

    def __init__(self, suite, number=None):
        if suite in self.__class__.SPECIALS:
            assert number is None
        else:
            assert suite in self.__class__.COLORS
            assert number in self.__class__.NUMBERS
        self.suite = suite
        self.number = number
        if self.number is not None:
            self.value = self.__class__.NUMBERS[self.number]
        else:
            self.value = None

    def __eq__(self, other):
        return self.suite == other.suite and self.number == other.number
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __lt__(self, other):
        if self.value is None:
            return False
        if other.value is None:
            return True
        return self.value < other.value

    def __str__(self):
        return self.suite + (str(self.number) if self.number else "")
    
    def __hash__(self):
        return(hash(str(self)))

class Deck:
    def __init__(self):
        self.cards = []
        for color in Card.COLORS:
            for number in Card.NUMBERS:
                self.cards.append(Card(color, number))
        for special in Card.SPECIALS:
            self.cards.append(Card(special))
        self.distributed_cards = []
    
    def distribute(self, seed=None):
        if seed is not None:
            random.seed(seed)
        shuffled = self.cards[:]
        random.shuffle(shuffled)
        num_cards = int(len(self.cards) / NUM_PLAYERS)
        return [shuffled[p*num_cards:(p+1)*num_cards] for p in range(NUM_PLAYERS)]
    
    def staged_distribute(self, num_cards, seed=None):
        if seed is None:
            random.seed(seed)
        shuffled = self.cards[:]
        random.shuffle(shuffled)
        
        card_set = set(self.cards)
        to_give = []
        for i in range(NUM_PLAYERS):
            given_card_set = set(self.distributed_cards)
            sampled_cards = random.sample(card_set - given_card_set, num_cards)
            self.distributed_cards += sampled_cards
            to_give.append(sampled_cards)
        return to_give

class Player:
    def __init__(self, game, player_id):
        self.game = game
        self.hand = []
        self.obtained = list()
        self.player_id = player_id
        self.card_locs = {}
        self.init_card_actions = []

    def remember_card_loc(self, player_id, card):
        self.card_locs[card] = player_id

    def possible_actions(self):
        assert self.game.turn == self.player_id
        if len(self.init_card_actions) == 0:
            self.init_card_actions = self.__class__.find_all_combinations(self.hand)
        return self.__class__._get_possible_actions(self.game, self.hand, self.init_card_actions)
    
    def add_cards_to_hand(self, cards):
        assert all(map(lambda x: isinstance(x, Card), cards))
        self.hand += cards
        self.card_locs.update({card: self.player_id for card in self.hand})

    @classmethod
    def find_all_combinations(cls, hand):
        '''Finds all combinations of given hand'''
        ## 'sort' by value
        card_counter = defaultdict(list)
        for card in hand:
            if card.number is None:
                card_counter[card.suite].append(card)
            else:
                card_counter[card.value].append(card)

        ## collect single-value combinations
        phoenix_list = card_counter["Phoenix"]
        assert len(phoenix_list) <= 1
        has_phoenix = len(phoenix_list) != 0
        num2comb = {1: Single, 2: Pair, 3: Triple, 4: FourCards}
        single_value_actions = []
        for card_value in card_counter.keys():
            if type(card_value) == str:
                continue # handle special cards later
            for num_cards in range(1, 5):
                for combination in combinations(card_counter[card_value], num_cards):
                    single_value_actions.append(num2comb[len(combination)](*combination))

                    if type(card_value) == int and len(combination) < 3 and has_phoenix:
                        # phoenix combinations
                        final_combination = combination + tuple(phoenix_list) # add phoenix if exists in hand
                        single_value_actions.append(num2comb[len(final_combination)](*final_combination))

        for card_value in filter(lambda x: type(x) == str and len(card_counter[x]) > 0, 
                                 card_counter.keys()): # special cards
            card = card_counter[card_value][0]
            if card == Card("MahJong"):
                for value in Card.NUMBERS.values():
                    single_value_actions.append(MahJongSingle(card, value))
            else:
                single_value_actions.append(Single(card))

        ## collect consecutive-value combinations
        # single-value consecutive sequence extraction
        seq_actions = []
        seq_dict = cls._get_seq_permutations(card_counter)

        # phoenix
        phoenix_seqs = []
        seq_ranges = sorted(list(seq_dict.keys()))
        if has_phoenix:
            for s_idx in range(len(seq_ranges)):
                backward_connectable = (s_idx > 0 and (seq_ranges[s_idx][0] - seq_ranges[s_idx-1][1]) == 2)
                forward_connectable = (s_idx < len(seq_ranges)-1 and (seq_ranges[s_idx+1][0] - seq_ranges[s_idx][1]) == 2)

                if forward_connectable:
                    for next_seq in seq_dict[seq_ranges[s_idx+1]]:
                        phoenix_seqs += [prev_seq+phoenix_list+next_seq for prev_seq in seq_dict[seq_ranges[s_idx]]]
                elif seq_ranges[s_idx][1] < 14:
                    phoenix_seqs += [seq+phoenix_list for seq in seq_dict[seq_ranges[s_idx]]]

                if backward_connectable:
                    pass # handled by forward connectable
                elif seq_ranges[s_idx][0] > 2:
                    phoenix_seqs += [phoenix_list+seq for seq in seq_dict[seq_ranges[s_idx]]]

        # all comb
        maximal_seqs = []
        if has_phoenix:
            maximal_seqs = phoenix_seqs
        else:
            maximal_seqs = sum(seq_dict.values(), [])

        straight_ables = cls._get_subsequences_over(maximal_seqs, 5)
        for seq in straight_ables:
            seq_actions.append(Straight(*seq))
            start_card_suite = seq[0].suite
            if all(map(lambda x: x.suite == start_card_suite, seq)):
                seq_actions.append(StraightFlush(*seq))

        # consecutive pairs
        all_pairs = filter(lambda x: isinstance(x, Pair), single_value_actions)
        pair_counter = defaultdict(list)
        for pair in all_pairs:
            pair_counter[pair.value].append(pair)

        pair_dict = cls._get_seq_permutations(pair_counter)
        pair_seqs = sum(pair_dict.values(), [])
        consecpair_ables = cls._get_subsequences_over(pair_seqs, 2)
        for seq in consecpair_ables:
            all_cards = sum((pair.cards for pair in seq), [])
            if len(set(all_cards)) != len(all_cards): # pheonix included twice
                phoenixes = list(filter(lambda c: c == Card("Phoenix"), all_cards))
                assert len(phoenixes) >= 2
                continue
            else:
                seq_actions.append(ConsecutivePair(*seq))

        ## Full House
        all_triples = filter(lambda x: type(x) == Triple, single_value_actions)
        full_houses = []
        for triple in all_triples:
            for pair in all_pairs:
                if triple.value != pair.value:
                    full_houses.append(FullHouse(pair, triple))

        return single_value_actions + seq_actions + full_houses
    
    @classmethod
    def _get_possible_actions(cls, game, hand, init_card_actions=None):
        actions = []
        curr_card_set = set(hand)
        if init_card_actions is None:
            init_card_actions = cls.find_all_combinations(hand)
        available_acts = filter(lambda x: len(set(x.cards) & curr_card_set) == len(x.cards), init_card_actions)
        if game.current:
            current_top = game.current[-1]
            actions = filter(lambda x: x.win(current_top), available_acts)
            if game.call_initiated and not game.call_satisifed:
                restricted_actions = filter(lambda x: x.value == game.call_value, actions) # includes singles and bombs
                restricted_actions = list(restricted_actions)
                if len(restricted_actions) != 0:
                    actions = restricted_actions
        else:
            if len(game.used) == 0:
                actions = filter(lambda x: isinstance(x, Single) and x.value == 1, 
                                 available_acts)
            else:
                actions = available_acts

        actions = list(actions)
        if (len(game.used) != 0 and 
            (((not game.call_initiated) or game.call_satisifed) or
             len(actions) == 0)):
            actions += [None]

        assert all([isinstance(action, Combination) or action is None for action in actions])
        return actions

    @classmethod
    def _get_seq_permutations(cls, counter):
        assert type(counter) == defaultdict
        seq_dict = dict()
        search_val = 2
        prev_seqs = [[]]
        while search_val < 15+1:
            val_cards = counter[search_val]
            if len(val_cards) == 0:
                if len(prev_seqs[0]) != 0:
                    card_values = list(map(lambda x: x.value, prev_seqs[0]))
                    min_v, max_v = min(card_values), max(card_values)
                    seq_dict[(min_v, max_v)] = prev_seqs
                prev_seqs = [[]]
            else:
                post_straights = []
                for card in val_cards:
                    post_straights += [s_comb+[card] for s_comb in prev_seqs]
                prev_seqs = post_straights
            search_val += 1
        return seq_dict
    
    @classmethod
    def _get_subsequences_over(cls, sequences, threshold):
        straight_seqs = filter(lambda x: len(x) >= threshold, sequences)
        subseq_list = []
        for seq in straight_seqs:
            for subseq_len in range(threshold, len(seq)+1):
                for start_idx in range(0, len(seq)+1-subseq_len):
                    subseq = seq[start_idx:start_idx+subseq_len]
                    assert len(subseq) >= threshold
                    subseq_list.append(subseq)
        return subseq_list

class Game:
    def __init__(self, seed=None, players=None):
        self.current = []
        self.pass_count = 0
        self.deck = Deck()
        if players is None:
            players = [Player for _ in range(NUM_PLAYERS)]
        else:
            assert len(players) == NUM_PLAYERS
        self.players = [p(self, i) for i, p in enumerate(players)]
        assert all(map(lambda x: isinstance(x, Player), self.players))
        self.turn = None
        self.exchange_index = np.identity(NUM_PLAYERS) - 1
        self.used = list()

        self.call_value = -1
        self.call_initiated = False
        self.call_satisifed = False
        self.called_big_tichu = [False for _ in range(4)]
        self.called_small_tichu = [False for _ in range(4)]

    def __str__(self):
        s = ""
        s += "Turn: {}\nPass Count: {}\n".format(self.turn, self.pass_count)
        s += "Exchange Index:\n{}\n".format(self.exchange_index)
        s += "Current\n"
        for i in range(len(self.current)):
            s += "- " + str(self.current[i]) + "\n"
        s += "Hands\n"
        for i in range(NUM_PLAYERS):
            s += "- player {}: {}\n".format(i, list(map(str, list(sorted(self.players[i].hand)))))
        s += "Obtained Cards\n"
        for i in range(NUM_PLAYERS):
            s += "- player {}: {}\n".format(i, list(map(str, self.players[i].obtained)))
        return s

    @property
    def unused_cards(self):
        return set(self.deck.cards) - set(self.used)

    def play(self, player, combi):
        assert self.turn == player
        next_turn = (self.turn + 1) % NUM_PLAYERS
        if combi:
            assert all([c in self.players[player].hand for c in combi.cards])
            if self.current:
                assert combi.win(self.current[-1])
            self.current.append(combi)
            if isinstance(combi, Single) and combi.card == Card("Dog"):
                # dog causes next player skip
                next_turn = (next_turn + 1) % NUM_PLAYERS
            if isinstance(combi, MahJongSingle):
                self.call_value = combi.call_value
                self.call_initiated = True

            elif (self.call_initiated and (not self.call_satisifed) and (combi.value == self.call_value)):
                self.call_satisifed = True

            for c in combi.cards:
                self.players[player].hand.remove(c)
            self.pass_count = 0
            for card in combi.cards:
                self.used.append(card)
                for player in self.players:
                    player.remember_card_loc(player, card)
        else:
            # pass
            self.pass_count += 1
            if self.pass_count == NUM_PLAYERS-1:
                self.players[next_turn].obtained += sum([ c.cards for c in self.current ], [])
                self.current = []
                self.pass_count = 0

        self.turn = next_turn
    
    def mark_exchange(self, giver, receiver, card_index):
        assert card_index in range(len(self.players[giver].hand))
        self.exchange_index[giver][receiver] = card_index

    def exchange(self):
        for i in range(NUM_PLAYERS):
            for j in range(NUM_PLAYERS):
                if i < j:
                    assert self.exchange_index[i,j] in range(len(self.players[i].hand))
                    assert self.exchange_index[j,i] in range(len(self.players[j].hand))
                    i_to_j = int(self.exchange_index[i,j])
                    j_to_i = int(self.exchange_index[j,i])

                    self.players[i].remember_card_loc(j, self.players[i].hand[i_to_j])
                    self.players[j].remember_card_loc(j, self.players[i].hand[i_to_j])
                    
                    self.players[i].remember_card_loc(i, self.players[j].hand[j_to_i])
                    self.players[j].remember_card_loc(i, self.players[j].hand[j_to_i])

                    # card swap
                    self.players[i].hand[i_to_j], self.players[j].hand[j_to_i] = self.players[j].hand[j_to_i], self.players[i].hand[i_to_j]

        for i in range(NUM_PLAYERS):
            if Card("MahJong") in self.players[i].hand:
                self.turn = i
                break
    
    @staticmethod
    def card_scorer(cards):
        score = 0
        for card in cards:
            if card.value == 10 or card.value == 13:
                score += 10
            elif card.value == 5:
                score += 5
            elif card.suite == "Dragon":
                score += 25
            elif card.suite == "Phoenix":
                score -= 25
        return score

    def run_game(self, upto='scoring', verbose=False):
        '''Runs game from beginning to a certain point.
        If runs to scoring, returns final scores of players.
        Otherwise, returns empty list.'''
        stages = ['bigTichu', 'exchange', 'firstRound', 'end', 'scoring']
        assert upto in stages
        upto_stage = stages.index(upto)
        
        scores = []
        if 0 <= upto_stage:
            # initial distribution of cards and extracting big tichu
            for idx, sampled_cards in enumerate(self.deck.staged_distribute(8)):
                self.players[idx].add_cards_to_hand(sampled_cards)
                self.called_big_tichu[idx] = self.players[idx].call_big_tichu()
        if 1 <= upto_stage:
            # exchanging
            
            for idx, sampled_cards in enumerate(self.deck.staged_distribute(6)):
                self.players[idx].add_cards_to_hand(sampled_cards) # distributing rest of cards
            
            exchange_pairs = []
            for p_idx in range(NUM_PLAYERS):
                for pair in self.players[p_idx].choose_exchange():
                    exchange_pairs.append((p_idx,)+pair)
            for exchange_v in exchange_pairs:
                self.mark_exchange(*exchange_v)
            self.exchange()
        if 2 <= upto_stage:
            # up to first round
            new_turn = True
            while new_turn or self.current:
                player = self.players[self.turn]
                a = player.sample_action()
                if verbose:
                    print(self.turn, a)
                self.play(self.turn, a)
                new_turn = False
        if 3 <= upto_stage:
            # rest of game 
            while True:
                player = self.players[self.turn]
                a = player.sample_action()
                if verbose:
                    print(self.turn, a)
                self.play(self.turn, a)
                left_cards = map(lambda x: int(len(x.hand) == 0), self.players)
                if sum(left_cards) >= 3:
                    break
        if 4 <= upto_stage:
            # scoring
            scores = [self.__class__.card_scorer(player.obtained) for player in self.players]

        return scores
        

class Bomb():
    pass

class Combination():
    
    """
    def __lt__(self, other):
        return self.value < other.value
    """

    def __eq__(self, other):
        if other is None:
            return False
        return set(self.cards) == set(other.cards)

    def __str__(self):
        return ",".join([str(c) for c in self.cards])

    def __len__(self):
        return len(self.cards)

    def update_value(self):
        pass
    
    def win(self, other):
        if isinstance(self, StraightFlush):
            if isinstance(other, StraightFlush):
                return len(self.cards) > len(other.cards) or self.value > other.value
            else:
                return True
        elif isinstance(self, FourCards):
            if isinstance(other, StraightFlush):
                return False
            elif isinstance(other, FourCards):
                return self.value > other.value
            else:
                return True
        elif isinstance(other, Bomb):
            # self not bomb, other is
            return False

        # dog stuff
        if isinstance(other, Single) and other.card == Card("Dog"):
            return True
        elif isinstance(self, Single) and self.card == Card("Dog"):
            return False

        if not (isinstance(self, other.__class__) or isinstance(other, self.__class__)):
            return False
        if len(self) != len(other):
            return False
        if isinstance(self, Single):
            self.update_value(other)
        else:
            self.update_value()
        return self.value > other.value

class Single(Combination):
    def __init__(self, card):
        assert isinstance(card, Card)
        self.cards = [card]

        if card == Card("MahJong"):
            self.value = 1
        elif card == Card("Dog"):
            self.value = 0
        elif card == Card("Dragon"):
            self.value = INF
        elif card == Card("Phoenix"):
            self.value = 1.5
        else:
            self.value = card.value

    @property
    def card(self):
        return self.cards[0]
    
    def update_value(self, current_top):
        if not current_top:
            pass
        if self.card == Card("Phoenix"):
            self.value = current_top.value + 0.5

class MahJongSingle(Single):
    def __init__(self, card, call_value):
        assert card == Card("MahJong")
        self.cards = [card]
        self.call_value = call_value
        self.value = 1

    def __str__(self):
        org_str = super().__str__()
        return org_str + f'; called {self.call_value}'


class Pair(Combination):
    def __init__(self, *cards):
        assert len(cards) == 2
        assert all([isinstance(c, Card) for c in cards])
        assert all([c.suite not in ["MahJong", "Dragon", "Dog"] for c in cards])
        not_phoenix = list(filter(lambda c: c != Card("Phoenix"), cards))
        assert len(set([c.value for c in not_phoenix])) == 1
        self.cards = list(cards)
        self.value = not_phoenix[0].value

class ConsecutivePair(Combination):
    def __init__(self, *pairs):
        assert all(isinstance(p, Pair) for p in pairs)
        pair_values = list([p.value for p in pairs])
        # check consecutive
        assert pair_values == list(range(min(pair_values), max(pair_values)+1))
        self.value = min(pair_values)
        self.cards = sum([p.cards for p in pairs], [])

class Triple(Combination):
    def __init__(self, *cards):
        assert len(cards) == 3
        assert all([isinstance(c, Card) for c in cards])
        assert all([c.suite not in ["MahJong", "Dragon", "Dog"] for c in cards])
        not_phoenix = list(filter(lambda c: c != Card("Phoenix"), cards))
        assert len(set([c.value for c in not_phoenix])) == 1
        self.cards = list(cards)
        self.value = not_phoenix[0].value

class FullHouse(Combination):
    def __init__(self, pair, triple):
        assert isinstance(pair, Pair) and isinstance(triple, Triple)
        assert pair.value != triple.value
        self.cards = pair.cards + triple.cards
        self.value = triple.value

class Straight(Combination):
    def __init__(self, *cards):
        assert len(cards) >= 5
        assert all([isinstance(c, Card) for c in cards])
        assert all([c.suite not in ["Dragon", "Dog"] for c in cards])

        first_card = cards[0]
        if cards[0] == Card("Phoenix"):
            self.value = cards[1].value-1
        elif cards[0] == Card("MahJong"):
            self.value = 1
        else:
            self.value = cards[0].value
        
        for i, c in enumerate(cards):
            if c == Card("MahJong"):
                assert i == 0
            else:
                assert c == Card("Phoenix") or c.value == self.value + i
        self.cards = list(cards)

# Bomb
class StraightFlush(Combination, Bomb):
     def __init__(self, *cards):
        assert len(cards) >= 5
        assert all([isinstance(c, Card) for c in cards])
        assert all([c.suite not in Card.SPECIALS for c in cards])
        card_values = list([c.value for c in cards])
        assert card_values == list(range(min(card_values), max(card_values)+1))
        card_suites = list([c.suite for c in cards])
        assert len(set(card_suites)) == 1
        self.cards = list(cards)
        self.value = min(card_values)

# Bomb
class FourCards(Combination, Bomb):
     def __init__(self, *cards):
        assert len(cards) == 4
        assert all([isinstance(c, Card) for c in cards])
        assert all([c.suite not in Card.SPECIALS for c in cards])
        card_values = list([c.value for c in cards])
        assert len(set(card_values)) == 1
        card_suites = list([c.suite for c in cards])
        assert len(set(card_suites)) == 4
        self.cards = cards
        self.value = min(card_values)

if __name__ == "__main__":
    game = Game(0)
    print(game)
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
    print(game)

    print(game.players[game.turn].possible_actions())
    """
    game.play(1, Single(Card("Phoenix")))
    game.play(2, None)
    game.play(3, None)
    game.play(0, None)
    pair2 = Pair(Card('Red', 2), Card('Black', 2))
    pair3 = Pair(Card('Red', 3), Card('Black', 3))
    game.play(1, pair2)
    game.play(2, pair3)
    game.play(3, None)
    game.play(0, None)
    game.play(1, None)
    triple1 = Triple(Card('Red', 2), Card('Black', 2), Card('Blue', 2))
    game.play(2, triple1)


    print(pair2, pair2.value)
    print(pair3, pair3.value)
    print(ConsecutivePair(pair2, pair3))
    print(Pair(Card('Red', 2), Card('Black', 2)))
    triple1 = Triple(Card('Red', 2), Card('Black', 2), Card('Blue', 2))
    print(Triple(Card('Red', 2), Card('Black', 2), Card('Blue', 2)).value)
    print(FullHouse(pair3, triple1))
    print(Straight(Card('MahJong'), Card('Black', 2), Card('Blue', 3), Card('Phoenix'), Card('Black', 5), Card('Blue', 6)))
    """
