"""actions.py — Action type definitions for the Sneak Peek Hold'em bot SDK.

This module defines the five legal action types that a bot may return from
``get_move()``.  All actions are implemented as lightweight ``namedtuple``
instances so they can be compared by identity (``isinstance`` checks in the
engine) and passed efficiently over the socket protocol.

Protocol Encoding:
    +---------------+--------+---------------------------------------------+
    | Action Class  | Code   | Description                                  |
    +===============+========+=============================================+
    | ActionFold    | ``F``  | Surrender the hand; forfeit the current pot. |
    +---------------+--------+---------------------------------------------+
    | ActionCall    | ``C``  | Match the opponent's current wager.          |
    +---------------+--------+---------------------------------------------+
    | ActionCheck   | ``K``  | Pass action when no bet is owed.             |
    +---------------+--------+---------------------------------------------+
    | ActionRaise   | ``R<n>``| Raise the total wager to ``n`` chips.        |
    +---------------+--------+---------------------------------------------+
    | ActionBid     | ``A<n>``| Submit a sealed auction bid of ``n`` chips.  |
    +---------------+--------+---------------------------------------------+

Note:
    ``ActionRaise`` and ``ActionBid`` each carry a single ``amount`` field
    (an integer number of chips).  The engine validates that ``amount`` falls
    within the legal raise/bid bounds before accepting the action.
"""
from collections import namedtuple

ActionFold = namedtuple('ActionFold', [])
"""Signal to fold the current hand.

No fields.  The engine immediately settles the pot in the opponent's favour
when a fold is received.
"""

ActionCall = namedtuple('ActionCall', [])
"""Signal to call (match) the opponent's outstanding bet.

No fields.  The engine deducts ``cost_to_call`` from the active player's stack
and advances to the next street if both players are now equal.
"""

ActionCheck = namedtuple('ActionCheck', [])
"""Signal to check (pass action) when no bet is owed.

No fields.  Only legal when ``cost_to_call == 0``.  Two consecutive checks
advance the game to the next street.
"""

# Bet & Raise is done through same action.
ActionRaise = namedtuple('ActionRaise', ['amount'])
"""Signal to bet or raise the total wager to a specified chip count.

Bet and Raise share this action type.  The engine distinguishes them
contextually: when the current wager is ``0`` it is a *bet*; otherwise it is a
*raise*.

Attributes:
    amount (int): The target *total* wager in chips (not the delta).  Must
        satisfy ``min_raise ≤ amount ≤ max_raise`` from ``raise_bounds``.
"""

ActionBid = namedtuple('ActionBid', ['amount'])
"""Signal to submit a sealed-bid auction offer.

Only legal during the ``'auction'`` street.  Both players submit their bids
simultaneously.  The engine resolves the second-price auction: the higher
bidder **pays the lower bid** and receives a randomly chosen opponent hole card.

Attributes:
    amount (int): Bid in chips.  Must satisfy ``0 ≤ amount ≤ my_chips``.
        A bid of ``0`` concedes the auction unconditionally.
"""