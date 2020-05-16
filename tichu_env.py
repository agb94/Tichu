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
    
    def __lt__(self, other):
        return self.number < other.number

    def __str__(self):
        return self.suite + (str(self.number) if self.number else "")

class Deck:
    def __init__(self):
        self.cards = []
        for color in Card.COLORS:
            for number in Card.NUMBERS:
                self.cards.append(Card(color, number))
        for special in Card.SPECIALS:
            self.cards.append(Card(special))
    
    def distribute(self, seed=None):
        if seed is not None:
            random.seed(seed)
        shuffled = self.cards[:]
        random.shuffle(shuffled)
        num_cards = len(self.cards) / NUM_PLAYERS
        return [shuffled[p*num_cards:(p+1)*num_cards] for p in range(NUM_PLAYERS)]

class Player:
    def __init__(self, game, player_id, hand):
        self.game = game
        self.hand = hand
        self.player_id = player_id

    def possible_actions(self):
        assert game.turn == self.player_id
        actions = []
        # ==========================================================================================
        if self.game.current:
            current_top = game.current[-1]
            card_counter = defaultdict(list)
            
            # 'sort' by value
            for card in self.hand:
                if card.number is None:
                    card_counter[card.suite].append(card)
                else:
                    card_counter[card.value].append(card)
            
            # collect single-value combinations
            for card_value in card_counter.keys():
                for num_cards in range(1, 5):
                    if card_value is None and num_cards != 1:
                        break # specials can't constitute combinations; exception handled later
                    for combination in combinations(card_counter[card_value], num_cards):
                        if num_cards == 1:
                            actions.append(Single(*combination))
                        elif num_cards == 2:
                            actions.append(Pair(*combination))
                        elif num_cards == 3:
                            actions.append(Triple(*combination))
                        elif num_cards == 4:
                            actions.append(FourCards(*combination))
        else:
            for card in self.hand:
                actions.append(Single(card))
        # ==========================================================================================
        assert all([isinstance(action, Combination) or action is None for action in actions])
        return actions

class Game:
    def __init__(self, seed=None):
        self.current = []
        self.pass_count = 0
        self.players = [Player(self, i, hand) for i, hand in enumerate(Deck().distribute(seed=seed))]
        self.turn = None
        self.obtained_cards = [list() for i in range(NUM_PLAYERS)]
        self.exchange_index = np.identity(NUM_PLAYERS) - 1

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
            s += "- player {}: {}\n".format(i, list(map(str, self.obtained_cards[i])))
        return s

    def play(self, player, combi):
        assert all([c in self.players[player].hand for c in combi.cards])
        assert self.turn == player
        next_turn = (self.turn + 1) % NUM_PLAYERS
        if combi:
            if self.current:
                assert combi.win(self.current[-1])
            self.current.append(combi)
            for c in combi.cards:
                self.players[player].hand.remove(c)
            self.pass_count = 0
        else:
            # pass
            self.pass_count += 1
            if self.pass_count == 3:
                self.obtained_cards[next_turn] += sum([ c.cards for c in self.current ], [])
                self.current = []
                self.pass_count = 0
        self.turn = (self.turn + 1) % NUM_PLAYERS
    
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
                    self.players[i].hand[i_to_j], self.players[j].hand[j_to_i] = self.players[j].hand[j_to_i], self.players[i].hand[i_to_j]
        for i in range(NUM_PLAYERS):
            if Card("MahJong") in self.players[i].hand:
                self.turn = i
                break

class Bomb():
    pass

class Combination():
    def __lt__(self, other):
        return self.value < other.value
    
    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        return ",".join([str(c) for c in self.cards])

    def __len__(self):
        return len(self.cards)

    def update_value(self):
        pass

    def win(self, other):
        if isinstance(self, Bomb):
            if isinstance(other, Bomb):
                # Both are bombs
                if self.__class__ != other.__class__:
                    return False
                else:
                    return self.value > other.value
            else:
                return True
        elif isinstance(other, Bomb):
            return False
        if self.__class__ != other.__class__:
            return False
        if len(self) != len(other):
            return False
        self.update_value(other)
        return self.value > other.value

class Single(Combination):
    def __init__(self, card):
        assert isinstance(card, Card)
        assert card != Card("Dog")
        self.cards = [card]

        if card == Card("MahJong"):
            self.value = 1
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
            self.value = cards[1]
        elif cards[0] == Card("MahJong"):
            self.value = 1
        else:
            self.value = cards[0].value
        
        for i, c in enumerate(cards):
            if c == Card("MahJong"):
                assert i == 0
            else:
                assert c == Card("Phoenix") or c.value == self.value + i
        self.cards = cards

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
        self.cards = cards
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

    #print(game.players[game.turn].possible_actions())
    game.play(3, Single(Card("MahJong")))
    game.play(0, Single(Card("Black", 2))) 
    print(game)
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
