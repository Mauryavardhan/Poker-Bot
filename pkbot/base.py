"""base.py — Abstract base class for all Sneak Peek Hold'em pokerbots.

This module defines ``BaseBot``, the mandatory interface that every bot
implementation must subclass and override.  The ``Runner`` in ``runner.py``
calls all three lifecycle hooks in the order shown below for each hand:

Lifecycle per hand::

    on_hand_start(game_info, current_state)
         │
         │  [get_move called 1+ times per street as the game progresses]
         ├──► get_move(game_info, current_state)  →  Action
         ├──► get_move(game_info, current_state)  →  Action
         │   ...
         │
    on_hand_end(game_info, current_state)

The default implementation of ``get_move`` included here is a minimal stub that
always calls/checks/bids 2.  Subclasses MUST override this method with a real
strategy.

Note:
    ``on_hand_start`` and ``on_hand_end`` raise ``NotImplementedError`` by
    default; subclasses must override them or they will crash at runtime.
"""
from .actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid
from .states import GameInfo, PokerState


class BaseBot():
    """Abstract base class that all pokerbots must inherit from.

    Provides three lifecycle hooks that the engine runner calls automatically.
    To create a functional bot, subclass ``BaseBot`` and override all three
    methods.  The class itself enforces the interface via ``NotImplementedError``
    on the two event hooks.

    Example::

        class MyBot(BaseBot):
            def on_hand_start(self, game_info, current_state):
                self.total_hands += 1

            def on_hand_end(self, game_info, current_state):
                print('Payoff:', current_state.payoff)

            def get_move(self, game_info, current_state):
                if current_state.street == 'auction':
                    return ActionBid(0)
                return ActionCheck() if current_state.cost_to_call == 0 else ActionFold()
    """

    def on_hand_start(self, game_info: GameInfo, current_state: PokerState) -> None:
        """Called exactly once at the beginning of each hand (round).

        Use this hook to reset per-hand state (e.g. clear flags, record
        starting chip counts) and to read the initial hole cards dealt to
        your bot.

        Args:
            game_info (GameInfo): Global game context.  Key fields:

                * ``bankroll`` (int) – cumulative chip delta from all previous
                  completed hands (positive = ahead, negative = behind).
                * ``time_bank`` (float) – remaining seconds in the global time
                  budget shared across all 1 000 rounds.
                * ``round_num`` (int) – current round number (1-indexed).

            current_state (PokerState): Snapshot of the table at hand start.
                Key fields at this point:

                * ``my_hand`` (list[str]) – your two hole cards, e.g. ``['Ah', 'Kd']``.
                * ``my_wager`` (int) – your posted blind (10 if SB, 20 if BB).
                * ``opp_wager`` (int) – opponent's posted blind.
                * ``my_chips`` (int) – chips remaining after the blind.

        Returns:
            None
        """
        raise NotImplementedError('on_hand_start')

    def on_hand_end(self, game_info: GameInfo, current_state: PokerState) -> None:
        """Called exactly once at the end of each hand, after all actions have resolved.

        Use this hook to update cross-hand statistics (opponent profiling,
        bankroll tracking, archetype classification, etc.).

        Args:
            game_info (GameInfo): Global game context, updated with the result
                of the hand just completed.  ``bankroll`` reflects the delta
                from this hand (added *before* this call).

            current_state (PokerState): Final table snapshot.  Key fields:

                * ``payoff`` (int) – your chip gain/loss for this hand.
                * ``street`` (str) – the street on which the hand ended.
                  ``'pre-flop'`` means a fold occurred before any community
                  cards were dealt; ``'river'`` means a showdown occurred.
                * ``my_wager`` / ``opp_wager`` (int) – final wagers; if equal,
                  the hand went to showdown.
                * ``opp_revealed_cards`` (list[str]) – opponent's revealed card
                  from the auction (if we won it), else ``[]``.

        Returns:
            None
        """
        raise NotImplementedError('on_hand_end')

    def get_move(self, game_info: GameInfo, current_state: PokerState) -> ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
        """Core decision method — called by the runner on every action request.

        This is where your bot's strategy lives.  The method must return a
        valid action within the 2-second per-query time limit.  The runner
        validates the returned action against ``current_state.legal_actions``
        and substitutes a ``CheckAction`` / ``FoldAction`` default if an
        illegal action is submitted.

        The stub implementation shown here is intentionally simplistic:
        it bids 2 in the auction and calls/checks passively otherwise.
        Override this method with a real strategy.

        Args:
            game_info (GameInfo): Global game context (bankroll, time_bank, round_num).
            current_state (PokerState): Full table snapshot.  Key fields:

                * ``street`` (str) – ``'pre-flop'``, ``'flop'``, ``'auction'``,
                  ``'turn'``, or ``'river'``.
                * ``cost_to_call`` (int) – chips required to stay in the hand.
                  ``0`` means you can check for free.
                * ``pot`` (int) – total chips in the pot (both players' invested chips).
                * ``my_chips`` / ``opp_chips`` (int) – remaining stacks.
                * ``raise_bounds`` (tuple[int, int]) – ``(min_raise, max_raise)``
                  for a legal raise.
                * ``can_act(ActionClass)`` – helper to test legal action membership.

        Returns:
            ActionFold | ActionCall | ActionCheck | ActionRaise | ActionBid:
                The chosen action for this decision point.
        """
        # raise NotImplementedError('get_move')
        print(current_state.street)
        if current_state.street == 'auction':
            return ActionBid(2)
        elif ActionCall in current_state.valid_actions:
            return ActionCall()

        elif ActionCheck in current_state.valid_actions:
            return ActionCheck()
        else:
            return ActionFold()