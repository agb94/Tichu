import random
import numpy as np
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
    
    def distribute(self):
        shuffled = self.cards[:]
        random.shuffle(shuffled)
        num_cards = len(self.cards) / NUM_PLAYERS
        return [list(sorted(shuffled[p*num_cards:(p+1)*num_cards])) for p in range(NUM_PLAYERS)]

class Game:
    def __init__(self):
        self.current = []
        self.pass_count = 0
        self.hands = tuple(Deck().distribute())
        self.turn = None
        self.obtained_cards = [list() for i in range(NUM_PLAYERS)]
        self.exchange_index = np.identity(NUM_PLAYERS) - 1

    def __str__(self):
        s = ""
        s += "Turn: {}\nPass Count: {}\n".format(self.turn, self.pass_count)
        s += "Exchange Index:\n{}\n".format(self.exchange_index)
        s += "Current\n"
        for i in range(len(self.current)):
            s += "-" + str(self.current[i])
        s += "Hands\n"
        for i in range(NUM_PLAYERS):
            s += "- player {}: {}\n".format(i, list(map(str, self.hands[i])))
        s += "Obtained Cards\n"
        for i in range(NUM_PLAYERS):
            s += "- player {}: {}\n".format(i, list(map(str, self.obtained_cards[i])))
        return s

    def play(self, player, combi):
        assert all([c in self.hands[player] for c in combi.cards])
        assert self.turn == player
        next_turn = (self.turn + 1) % 4
        if combi:
            if self.current:
                assert self.current[-1].__class__ == combi.__class__
                if isinstance(combi, Single) and combi.card == Card("Phoenix"):
                    combi.value = self.current[-1].value + 0.5
                assert combi.value > self.current[-1].value
            self.current.append(combi)
            for c in combi.cards:
                self.hands[player].remove(c)
            self.pass_count = 0
        else:
            # pass
            self.pass_count += 1
            if self.pass_count == 3:
                self.obtained_cards[next_turn] += sum([ c.cards for c in self.current ], [])
                self.current = []
                self.pass_count = 0
        self.turn = (self.turn + 1) % 4
    
    def mark_exchange(self, giver, receiver, card_index):
        assert card_index in range(len(self.hands[giver]))
        self.exchange_index[giver][receiver] = card_index

    def exchange(self):
        for i in range(NUM_PLAYERS):
            for j in range(NUM_PLAYERS):
                if i < j:
                    assert self.exchange_index[i,j] in range(len(self.hands[i]))
                    assert self.exchange_index[j,i] in range(len(self.hands[j]))
                    i_to_j = int(self.exchange_index[i,j])
                    j_to_i = int(self.exchange_index[j,i])
                    self.hands[i][i_to_j], self.hands[j][j_to_i] = self.hands[j][j_to_i], self.hands[i][i_to_j]
        for i in range(NUM_PLAYERS):
            if Card("MahJong") in self.hands[i]:
                self.turn = i
                break

class Combination():
    def __lt__(self, other):
        return self.value < other.value
    
    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        return ",".join([str(c) for c in self.cards])

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

if __name__ == "__main__":
    game = Game()
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

    """
    game.play(0, Single(Card("Black", 3)))
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
