"""states.py — Game state data structures for the Sneak Peek Hold'em bot SDK.

This module defines the data-model layer that the ``Runner`` uses to reconstruct
the game tree locally from the action history transmitted by the engine.  Bots
interact with these objects through the ``PokerState`` facade; the internal
``GameState`` immutable tree is an implementation detail.

Module-level Game Constants:
    NUM_ROUNDS (int): Total hands per match — 1 000.
    STARTING_STACK (int): Chips each player starts each hand with — 5 000.
    BIG_BLIND (int): Big blind size — 20 chips.
    SMALL_BLIND (int): Small blind size — 10 chips.

Class Hierarchy:
    * ``GameInfo`` (namedtuple) — Immutable per-query global metadata.
    * ``HandResult`` (namedtuple) — Terminal node carrying payoffs.
    * ``GameState`` (namedtuple subclass) — Immutable game-tree node.
    * ``PokerState`` — Mutable, user-facing wrapper over ``GameState``/``HandResult``.

Street Encoding (``GameState.street`` integer field):
    +-------+-------------+
    | Value | Street      |
    +=======+=============+
    | 0     | Pre-flop    |
    +-------+-------------+
    | 3     | Flop        |
    +-------+-------------+
    | 4     | Turn        |
    +-------+-------------+
    | 5     | River       |
    +-------+-------------+
    | *     | auction=True|
    +-------+-------------+

    When ``GameState.auction`` is ``True``, ``get_street_name()`` returns
    ``'auction'`` regardless of the ``street`` integer.
"""
from collections import namedtuple
from .actions import ActionFold, ActionCall, ActionCheck, ActionRaise, ActionBid

GameInfo = namedtuple('GameInfo', ['bankroll', 'time_bank', 'round_num'])
"""Immutable snapshot of global game metadata passed to every bot lifecycle hook.

Attributes:
    bankroll (int): Cumulative chip delta from the start of the match to the
        *start* of the current hand.  Positive = winning; negative = losing.
    time_bank (float): Remaining seconds in the bot's global time budget shared
        across all rounds.  Starts at 30.0 s and decreases with each query.
    round_num (int): Current hand number, 1-indexed (1 … NUM_ROUNDS).
"""

HandResult = namedtuple('HandResult', ['payoffs', 'bids', 'parent_state'])
"""Terminal game-tree node produced when a hand concludes (fold or showdown).

Attributes:
    payoffs (list[int, int]): Net chip change for each player (index 0 = SB,
        index 1 = BB).  Values are equal and opposite in a zero-sum game.
    bids (list[int | None, int | None]): Final auction bids submitted by each
        player.  ``None`` if the auction phase was not reached.
    parent_state (GameState): The ``GameState`` immediately preceding the
        terminal event, used by ``PokerState`` to expose board/hand information
        after the hand is over.
"""

NUM_ROUNDS = 1000
STARTING_STACK = 5000
BIG_BLIND = 20
SMALL_BLIND = 10


class GameState(namedtuple('_GameState', ['dealer', 'street', 'auction', 'bids', 'wagers', 'chips', 'hands', 'opp_hands', 'community_cards', 'parent_state'])):
    """Immutable node in the game tree representing one decision point in a hand.

    ``GameState`` is a namedtuple subclass whose fields completely encode the
    table at a specific moment.  All state transitions return a **new**
    ``GameState`` (or ``HandResult``) rather than mutating the existing one,
    making it safe to save and backtrack through the game tree.

    Attributes:
        dealer (int): Monotonically increasing action counter.  Even values
            indicate it is the SB's turn; odd values indicate BB's turn.
            ``dealer % 2`` gives the index of the **active** player.
        street (int): Integer encoding of the current betting round (0, 3, 4, 5).
            See module docstring for the encoding table.
        auction (bool): ``True`` during the sealed-bid auction phase after the
            flop is dealt.  When ``True``, the only legal action is ``ActionBid``.
        bids (list[int | None, int | None]): Mutable list tracking each player's
            submitted auction bid.  ``None`` means the player has not yet bid.
        wagers (list[int, int]): Total chips committed to the pot by each player
            in the **current betting round**.  Resets to ``[0, 0]`` each street.
        chips (list[int, int]): Remaining chip counts for each player.
        hands (list[list[str], list[str]]): Each player's private hole cards as
            strings (e.g. ``['Ah', 'Kd']``).  Index 0 = SB, index 1 = BB.
        opp_hands (list[list[str], list[str]]): Opponent cards revealed via the
            auction win.  Usually ``[[], []]``; populated after auction resolves.
        community_cards (list[str]): Board cards visible to both players.
            Empty list pre-flop; grows to 3, 4, 5 cards through the streets.
        parent_state (GameState | None): Previous ``GameState`` in the game tree;
            ``None`` at the root (hand start).
    """

    def get_street_name(self):
        """Return the human-readable name of the current street.

        Checks ``self.auction`` first because the auction occurs mid-street
        (the ``street`` integer is still 3 during the auction).

        Returns:
            str: One of ``'pre-flop'``, ``'flop'``, ``'auction'``,
                ``'turn'``, or ``'river'``.
        """
        if self.auction:
            return 'auction'
        return {
            0: 'pre-flop',
            3: 'flop',
            4: 'turn',
            5: 'river'
        }[self.street]

    def calculate_result(self):
        """Compare the players' hands and compute chip payoffs.

        Only called by ``next_street()`` after the river (``street == 5``).
        In the SDK's local ``states.py`` (used for tree reconstruction by the
        runner), showdown evaluation always returns ``[0, 0]`` payoffs because
        the actual hand comparison is performed by the authoritative engine
        process — the runner receives the real result via the ``'D<delta>'``
        message and updates payoffs accordingly.

        Returns:
            HandResult: Terminal node with ``payoffs = [0, 0]``, the bids
                list, and a reference to this ``GameState`` as ``parent_state``.
        """
        return HandResult([0, 0], self.bids, self)

    def get_valid_actions(self):
        """Return the set of action classes available to the active player.

        Logic:
            * During **auction**: only ``{ActionBid}`` is legal.
            * When **no bet is owed** (``cost == 0``): ``{ActionCheck}`` always,
              plus ``{ActionRaise}`` unless either player is all-in.
            * When **a bet is owed** (``cost > 0``): ``{ActionFold, ActionCall}``
              always, plus ``{ActionRaise}`` if neither player is all-in and the
              active player has enough chips to re-raise.

        Returns:
            set[type]: A set of action *classes* (not instances).
        """
        if self.auction:
            return {ActionBid}
        active_idx = self.dealer % 2
        cost = self.wagers[1-active_idx] - self.wagers[active_idx]
        if cost == 0:
            # we can only raise the stakes if both players can afford it
            cannot_bet = (self.chips[0] == 0 or self.chips[1] == 0)
            return {ActionCheck} if cannot_bet else {ActionCheck, ActionRaise}
        # cost > 0
        # similarly, re-raising is only allowed if both players can afford it
        cannot_raise = (cost == self.chips[active_idx] or self.chips[1-active_idx] == 0)
        return {ActionFold, ActionCall} if cannot_raise else {ActionFold, ActionCall, ActionRaise}

    def get_raise_limits(self):
        """Compute the legal raise amount boundaries for the active player.

        The minimum raise is at least the size of the previous bet/raise
        (``max(cost, BIG_BLIND)``), capped at the effective stack.  The maximum
        raise is capped at the smaller of the two players' remaining chips to
        prevent raising more than the opponent can call.

        Returns:
            tuple[int, int]: ``(min_total_wager, max_total_wager)`` where each
                value is the **total** wager (``wagers[active] + raise_amount``),
                not the incremental raise size.  Use these directly as the
                ``amount`` in ``ActionRaise(amount)``.
        """
        active_idx = self.dealer % 2
        cost = self.wagers[1-active_idx] - self.wagers[active_idx]
        max_bet = min(self.chips[active_idx], self.chips[1-active_idx] + cost)
        min_bet = min(max_bet, cost + max(cost, BIG_BLIND))
        return (self.wagers[active_idx] + min_bet, self.wagers[active_idx] + max_bet)

    def next_street(self):
        """Advance the game tree to the next betting round.

        Called automatically after ``ActionCall`` (when both wagers equalise)
        and after two consecutive ``ActionCheck`` actions.

        Transition logic:
            * ``street == 5`` (river) → ``calculate_result()`` (showdown).
            * ``street == 0`` (pre-flop) → auction phase (``auction=True``, wagers reset).
            * Otherwise → next street (``street + 1``), wagers reset to ``[0, 0]``.

        Returns:
            GameState | HandResult: The next node in the game tree.
        """
        if self.street == 5:
            return self.calculate_result()
        if self.street == 0:
            return GameState(1, 3, True, self.bids, [0, 0], self.chips, self.hands, self.opp_hands, self.community_cards, self)
        return GameState(1, self.street+1, False, self.bids, [0, 0], self.chips, self.hands, self.opp_hands, self.community_cards, self)

    def apply_action(self, action):
        """Transition the game tree by applying one player action.

        Handles all five action types and returns the resulting state.  The
        method is pure (no side effects) — it always returns a new node.

        Action semantics:
            * ``ActionFold``: Immediately terminates the hand.  The folding
              player forfeits their invested chips; payoff is computed from the
              chip delta relative to ``STARTING_STACK``.
            * ``ActionCall``: Matches the opponent's wager.  Special case: SB
              calling BB pre-flop (``dealer == 0``) resets wagers to the BB
              level before advancing to the next street.
            * ``ActionCheck``: Passes action.  Two consecutive checks (or a
              check after all streets are done) advances to the next street.
            * ``ActionBid``: Records the bid.  When both bids are set,
              resolves the second-price auction (in the runner's SDK copy,
              the actual chip transfer and card reveal are handled by the
              authoritative engine; here only the state transition is tracked).
            * ``ActionRaise``: Updates the active player's wager to
              ``action.amount`` and decrements their chip count accordingly.

        Args:
            action (ActionFold | ActionCall | ActionCheck | ActionBid | ActionRaise):
                The action to apply.

        Returns:
            GameState | HandResult: The next game-tree node.
        """
        active = self.dealer % 2
        if isinstance(action, ActionFold):
            delta = self.chips[0] - STARTING_STACK if active == 0 else STARTING_STACK - self.chips[1]
            return HandResult([delta, -delta], self.bids, self)
        if isinstance(action, ActionCall):
            if self.dealer == 0:  # sb calls bb
                return GameState(1, 0, self.auction, self.bids, [BIG_BLIND] * 2, [STARTING_STACK - BIG_BLIND] * 2, self.hands, self.opp_hands, self.community_cards, self)
            # match bet
            next_wagers = list(self.wagers)
            next_chips = list(self.chips)
            amt = next_wagers[1-active] - next_wagers[active]
            next_chips[active] -= amt
            next_wagers[active] += amt
            state = GameState(self.dealer + 1, self.street, self.auction, self.bids, next_wagers, next_chips, self.hands, self.opp_hands, self.community_cards, self)
            return state.next_street()
        if isinstance(action, ActionCheck):
            if (self.street == 0 and self.dealer > 0) or self.dealer > 1:  # both players acted
                return self.next_street()
            # check
            return GameState(self.dealer + 1, self.street, self.auction, self.bids, self.wagers, self.chips, self.hands, self.opp_hands, self.community_cards, self)
        
        if isinstance(action, ActionBid):
            self.bids[active] = -1
            if None not in self.bids: 
                if self.bids[0] == self.bids[1]:
                    state = GameState(1, self.street, False, self.bids, self.wagers, self.chips, self.hands, self.opp_hands, self.community_cards, self)

                else:
                    state = GameState(1, self.street, False, self.bids, self.wagers, self.chips, self.hands, self.opp_hands, self.community_cards, self)
                return state
            
            else:
                return GameState(self.dealer + 1, self.street, self.auction, self.bids, self.wagers, self.chips, self.hands, self.opp_hands, self.community_cards, self)
        # isinstance(action, ActionRaise)
        next_wagers = list(self.wagers)
        next_chips = list(self.chips)
        added = action.amount - next_wagers[active]
        next_chips[active] -= added
        next_wagers[active] += added
        return GameState(self.dealer + 1, self.street, self.auction, self.bids, next_wagers, next_chips, self.hands, self.opp_hands, self.community_cards, self)


class PokerState:
    """User-facing facade over ``GameState`` / ``HandResult`` exposing clean, flat attributes.

    ``PokerState`` is the object passed to all three bot lifecycle hooks.  It
    translates the internal game-tree representation into a flat, intuitive API
    that bots interact with directly.  Its attributes are set once in ``__init__``
    and treated as read-only by convention.

    Attributes:
        is_terminal (bool): ``True`` if the hand has ended (fold or showdown).
            When ``True``, ``legal_actions`` is empty and ``payoff`` is set.
        street (str): Current street name — ``'pre-flop'``, ``'flop'``,
            ``'auction'``, ``'turn'``, or ``'river'``.
        my_hand (list[str]): Your two private hole cards, e.g. ``['Ah', 'Kd']``.
        board (list[str]): Community cards visible to both players (0–5 cards).
        opp_revealed_cards (list[str]): Opponent hole card(s) revealed via
            winning the auction.  Empty list if you did not win the auction.
        my_chips (int): Your remaining chip count (not counting wager already in pot).
        opp_chips (int): Opponent's remaining chip count.
        my_wager (int): Chips you have committed to the pot this street.
        opp_wager (int): Chips opponent has committed to the pot this street.
        pot (int): Total chips invested by both players across all streets:
            ``(STARTING_STACK − my_chips) + (STARTING_STACK − opp_chips)``.
        cost_to_call (int): Additional chips needed to stay in the hand
            (``opp_wager − my_wager``).  ``0`` when you can check for free.
        is_bb (bool): ``True`` if you are the Big Blind this hand.
        legal_actions (set[type]): Action classes currently available.
            Use ``can_act(ActionClass)`` for convenience.
        payoff (int): Your chip gain/loss for this hand.  Only meaningful when
            ``is_terminal == True``; ``0`` otherwise.
        raise_bounds (tuple[int, int]): ``(min_raise, max_raise)`` total wager
            amounts for a legal raise.  ``(0, 0)`` when raising is not allowed.
    """
    is_terminal: bool
    street: str
    my_hand: list[str]
    board: list[str]
    opp_revealed_cards: list[str]
    my_chips: int
    opp_chips: int
    my_wager: int
    opp_wager: int
    pot: int
    cost_to_call: int
    is_bb: bool
    legal_actions: set
    payoff: int
    raise_bounds: tuple[int, int]

    def __init__(self, state, active):
        """Construct a ``PokerState`` from an internal game-tree node.

        Reads all relevant information from either a ``GameState`` (mid-hand)
        or a ``HandResult`` (terminal) and exposes it through flat attributes.
        When the state is terminal, ``parent_state`` is consulted for board and
        hand information since ``HandResult`` itself does not carry those fields.

        Args:
            state (GameState | HandResult): The current game-tree node.
            active (int): Index of the bot reading this state (``0`` = SB,
                ``1`` = BB).  Used to separate ``my_*`` from ``opp_*`` fields.
        """
        self.is_terminal = isinstance(state, HandResult)
        # If terminal, we look at the parent state for the board/hands info
        current_state = state.parent_state if self.is_terminal else state

        self.street = current_state.get_street_name() # 'Pre-Flop', 'Flop', 'Auction', 'Turn', or 'River'
        self.my_hand = current_state.hands[active]
        self.board = current_state.community_cards
        self.opp_revealed_cards = current_state.opp_hands[active]
        
        self.my_chips = current_state.chips[active]
        self.opp_chips = current_state.chips[1-active]
        self.my_wager = current_state.wagers[active]
        self.opp_wager = current_state.wagers[1-active]
        
        self.pot = (STARTING_STACK - self.my_chips) + (STARTING_STACK - self.opp_chips)
        self.cost_to_call = self.opp_wager - self.my_wager
        self.is_bb = active == 1
        
        if self.is_terminal:
            self.legal_actions = set()
            self.payoff = state.payoffs[active]
            self.raise_bounds = (0, 0)
        else:
            self.legal_actions = current_state.get_valid_actions()
            self.payoff = 0
            self.raise_bounds = current_state.get_raise_limits()

    def can_act(self, action_cls):
        """Check whether a specific action class is currently legal.

        A convenience wrapper around ``action_cls in self.legal_actions`` that
        reads more naturally in decision-making code.

        Args:
            action_cls (type): One of ``ActionFold``, ``ActionCall``,
                ``ActionCheck``, ``ActionRaise``, or ``ActionBid``.

        Returns:
            bool: ``True`` if the action class is in ``self.legal_actions``.

        Example::

            if current_state.can_act(ActionRaise):
                min_r, max_r = current_state.raise_bounds
                return ActionRaise(min_r)
        """
        return action_cls in self.legal_actions