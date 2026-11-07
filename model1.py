import random

from card_utils import Card, evaluate, full_deck
from player_action import PlayerAction


class Model:

    def __init__(self):
        self.hand = []
        self.player_id = None
        self.hands_played = 0

    def seed(self, hand, player_id):
        self.hand = list(hand)
        self.player_id = player_id
        self.hands_played += 1

    def calculate_equity(self, game_state, simulations=140):

        me = self.player_id
        board = list(game_state.board)

        opponents = [
            p for p in game_state.players_in_hand
            if p != me
        ]

        if not opponents:
            return 1.0

        known = self.hand + board

        deck = [
            c for c in full_deck()
            if c not in known
        ]

        missing = 5 - len(board)

        seed = sum(c.value * (i + 1)
                   for i, c in enumerate(self.hand + board))

        seed += game_state.pot * 17
        seed += self.hands_played * 31

        rng = random.Random(seed)

        equity = 0.0

        for _ in range(simulations):

            cards_needed = len(opponents) * 2 + missing

            drawn = rng.sample(deck, cards_needed)

            pos = 0

            opponent_hands = []

            for _ in opponents:
                opponent_hands.append(drawn[pos:pos + 2])
                pos += 2

            future_board = board + drawn[pos:]

            my_rank = evaluate(self.hand + future_board)

            opponent_ranks = [
                evaluate(hand + future_board)
                for hand in opponent_hands
            ]

            best = max([my_rank] + opponent_ranks)

            winners = (
                (1 if my_rank == best else 0)
                + sum(1 for r in opponent_ranks if r == best)
            )

            if my_rank == best:
                equity += 1.0 / winners

        return equity / simulations

    def run(self, actions, game_state):

        me = self.player_id

        call_amount = game_state.call_amount
        pot = max(game_state.pot, 1)

        # Estimate hand strength.
        equity = self.calculate_equity(game_state)

        pot_odds = game_state.pot_odds

        # Preflop strategy.
        if game_state.street == "preflop":

            values = sorted(
                [c.value for c in self.hand],
                reverse=True
            )

            high, low = values

            suited = self.hand[0].suit == self.hand[1].suit
            pair = high == low
            gap = high - low

            premium = (
                pair and high >= 10
            ) or (
                high == 14 and low >= 12
            )

            playable = (
                pair
                or high >= 13
                or (high >= 11 and low >= 9)
                or (suited and high >= 10)
            )

            if premium:
                equity = max(equity, 0.72)

            elif playable:
                equity = max(equity, 0.55)

            elif high >= 10 and gap <= 2:
                equity = max(equity, 0.48)

        # Decide whether to raise.
        if game_state.can_raise:

            should_raise = False

            if game_state.street == "preflop":

                values = sorted(
                    [c.value for c in self.hand],
                    reverse=True
                )

                high, low = values

                if high == low and high >= 10:
                    should_raise = True

                elif high == 14 and low >= 11:
                    should_raise = True

                elif high >= 13 and low >= 10:
                    should_raise = True

            else:

                if equity > max(0.68, pot_odds + 0.28):
                    should_raise = True

            if should_raise:

                minimum, maximum = game_state.raise_range

                if game_state.street == "preflop":
                    target = minimum
                else:
                    target = int(
                        game_state.bets[me] + pot * 0.65
                    )

                target = max(minimum, min(maximum, target))

                return PlayerAction.raise_to(me, int(target))

        # Nothing to call: check.
        if call_amount == 0:
            return PlayerAction.call(me)

        # Compare estimated equity with pot odds.
        if equity > pot_odds + 0.07:
            return PlayerAction.call(me)

        # Avoid expensive marginal calls.
        return PlayerAction.fold(me)

    def end_game(self, actions, game_state):
        pass
