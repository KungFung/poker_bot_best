from card_utils import Card
from game_state import GameStateView
from player_action import PlayerAction


class Model:

    def __init__(self):
        self.hand = []
        self.player_id = None

    def seed(self, hand: list[Card], player_id: int):
        self.hand = hand
        self.player_id = player_id

    def run(
        self,
        actions: list[PlayerAction],
        game_state: GameStateView
    ) -> PlayerAction:

        me = self.player_id

        # ALWAYS GO ALL-IN WHEN A RAISE IS AVAILABLE
        if game_state.can_raise:
            min_raise, max_raise = game_state.raise_range

            return PlayerAction.raise_to(
                me,
                max_raise
            )

        # If raising is impossible, never fold.
        # Call or check instead.
        return PlayerAction.call(me)

    def end_game(
        self,
        actions: list[PlayerAction],
        game_state: GameStateView
    ):
        pass
