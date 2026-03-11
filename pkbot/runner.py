"""runner.py — Socket communication layer between bots and the Sneak Peek Hold'em engine.

This module provides the ``Runner`` class (responsible for parsing engine messages
and reconstructing the game tree locally) and two public helper functions
(``parse_args`` and ``run_bot``) that form the standard entry point for every
bot script.

Communication Protocol:
    The engine and each bot communicate over a **line-delimited TCP socket**.
    Each engine message is a space-separated sequence of *clauses*, where the
    first character of each clause identifies its type:

    +------+-------------------------------------------------------+
    | Code | Meaning                                               |
    +======+=======================================================+
    | ``T``| Time remaining in the bot's global budget (float)    |
    +------+-------------------------------------------------------+
    | ``P``| Player index (``0`` = SB, ``1`` = BB)                |
    +------+-------------------------------------------------------+
    | ``H``| Hole cards dealt, comma-separated (e.g. ``Ah,Kd``)   |
    +------+-------------------------------------------------------+
    | ``B``| Board cards, comma-separated (e.g. ``Jh,2c,9s``)    |
    +------+-------------------------------------------------------+
    | ``N``| Post-auction state: ``chips_bid-ratio_revealed_card`` |
    +------+-------------------------------------------------------+
    | ``O``| Opponent showdown reveal (comma-separated cards)      |
    +------+-------------------------------------------------------+
    | ``D``| Round delta (chip gain/loss for this player)          |
    +------+-------------------------------------------------------+
    | ``F``| Opponent folded                                       |
    +------+-------------------------------------------------------+
    | ``C``| Opponent called                                       |
    +------+-------------------------------------------------------+
    | ``K``| Opponent checked                                      |
    +------+-------------------------------------------------------+
    | ``R``| Opponent raised to amount (e.g. ``R120``)             |
    +------+-------------------------------------------------------+
    | ``A``| Opponent bid amount (e.g. ``A50``)                    |
    +------+-------------------------------------------------------+
    | ``Q``| Quit signal — engine is shutting down                 |
    +------+-------------------------------------------------------+

Bot responses use the same single-character action codes (``F``, ``C``, ``K``,
``R<n>``, ``A<n>``).  An engine-side ``K`` (check) is sent by the runner back
to the engine as an acknowledgement at the end of each round.

Typical Usage::

    from pkbot.runner import parse_args, run_bot
    from mybot import Player

    if __name__ == '__main__':
        run_bot(Player(), parse_args())
"""
import argparse
import socket
from .actions import ActionBid, ActionFold, ActionCall, ActionCheck, ActionRaise
from .states import GameInfo, HandResult, GameState, PokerState
from .states import STARTING_STACK, BIG_BLIND, SMALL_BLIND
from .base import BaseBot


class Runner():
    """Manages the game loop for a single bot over a socket connection to the engine.

    ``Runner`` owns the socket file handle and translates the engine's text
    protocol into ``GameState`` / ``PokerState`` objects, calling the bot's
    lifecycle hooks at the appropriate moments.  It also serialises the bot's
    ``Action`` objects back into the wire format.

    The internal game tree is reconstructed locally by replaying each action
    clause received from the engine.  This means the local ``GameState`` always
    mirrors the authoritative engine state (within the information available to
    this player).

    Attributes:
        pokerbot (BaseBot): The bot instance whose lifecycle hooks are called.
        socketfile: A read/write file handle wrapping the TCP socket connection
            to the engine (created via ``socket.makefile('rw')``).
    """

    def __init__(self, pokerbot, socketfile):
        """Initialise the runner with a bot instance and an open socket file.

        Args:
            pokerbot (BaseBot): A fully initialised bot implementing the
                ``BaseBot`` interface.
            socketfile: A read/write text-mode file wrapping the TCP socket
                to the engine.  Typically obtained via ``sock.makefile('rw')``.
        """
        self.pokerbot = pokerbot
        self.socketfile = socketfile

    def receive(self):
        """Yield incoming engine message packets as lists of clause strings.

        Reads lines from the socket file in a blocking loop, splits each line
        on whitespace, and yields the resulting list of clauses.  Terminates
        when the socket is closed or an empty packet is received.

        Yields:
            list[str]: One packet (list of clause strings) per engine message.
                Each clause starts with a single-character type code (see module
                docstring for the full protocol table).
        """
        while True:
            packet = self.socketfile.readline().strip().split(' ')
            if not packet:
                break
            yield packet

    def send(self, action):
        """Encode an action and write it to the engine socket.

        Translates an ``Action`` namedtuple into the single-character wire
        format, appends a newline, and flushes the buffer.

        Wire encoding:
            * ``ActionFold``  → ``'F'``
            * ``ActionCall``  → ``'C'``
            * ``ActionCheck`` → ``'K'``
            * ``ActionBid(n)``→ ``'A<n>'``  (e.g. ``'A50'``)
            * ``ActionRaise(n)``→``'R<n>'`` (e.g. ``'R120'``)

        Args:
            action (ActionFold | ActionCall | ActionCheck | ActionBid | ActionRaise):
                The action to transmit.
        """
        if isinstance(action, ActionFold):
            code = 'F'
        elif isinstance(action, ActionCall):
            code = 'C'
        elif isinstance(action, ActionCheck):
            code = 'K'
        elif isinstance(action, ActionBid): 
            code = 'A' + str(action.amount)
        else:  # isinstance(action, ActionRaise)
            code = 'R' + str(action.amount)
        self.socketfile.write(code + '\n')
        self.socketfile.flush()

    def run(self):
        """Main event loop — drives the bot through an entire 1 000-round match.

        Reads packets from the engine, parses each clause to update the local
        game-tree state, and calls the bot's lifecycle hooks at the correct
        moments.  The protocol state machine handles the following clause types:

        * **``T``**: Updates the time bank in ``game_info``.
        * **``P``**: Sets the active player index.
        * **``H``**: Deals hole cards, initialises ``GameState``, and calls
          ``on_hand_start`` (once per hand via ``round_flag``).
        * **``F/C/K/R/A``**: Applies opponent actions to the local game tree.
        * **``N``**: Post-auction update — injects new chip counts, bid values,
          and the revealed opponent card into the state.
        * **``B``**: Deals board (community) cards at each new street.
        * **``O``**: Showdown reveal — backtracks the state to inject the
          opponent's hole cards for transparency.
        * **``D``**: Round payoff — updates ``game_info.bankroll``, calls
          ``on_hand_end``, then increments ``round_num``.
        * **``Q``**: Quit signal — exits the loop cleanly.

        Between clauses, the runner invokes ``pokerbot.get_move()`` whenever it
        is our turn to act (``active == state.dealer % 2``) and sends the
        returned action back to the engine.

        Returns:
            None: Returns when the ``'Q'`` quit signal is received or the
                socket connection is closed.
        """
        game_info = GameInfo(0, 0., 1)
        state: GameState = None
        active = 0
        round_flag = True
        for packet in self.receive():
            for clause in packet:
                if clause[0] == 'T':
                    game_info = GameInfo(game_info.bankroll, float(clause[1:]), game_info.round_num)
                elif clause[0] == 'P':
                    active = int(clause[1:])
                elif clause[0] == 'H':
                    hands = [[], []]
                    hands[active] = clause[1:].split(',')
                    wagers = [SMALL_BLIND, BIG_BLIND]
                    chips = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
                    state = GameState(0, 0, False, [None, None], wagers, chips, hands, [[], []], [], None)
                    if round_flag:
                        self.pokerbot.on_hand_start(game_info, PokerState(state, active))
                        round_flag = False
                elif clause[0] == 'F':
                    state = state.apply_action(ActionFold())
                elif clause[0] == 'C':
                    state = state.apply_action(ActionCall())
                elif clause[0] == 'K':
                    state = state.apply_action(ActionCheck())
                elif clause[0] == 'R':
                    state = state.apply_action(ActionRaise(int(clause[1:])))
                elif clause[0] == 'A': 
                    state = state.apply_action(ActionBid(int(clause[1:])))
                elif clause[0] == 'N':
                    # Post-auction notification:
                    # Format: N<chips0>,<chips1>_<bid0>,<bid1>_<revealed_card1>,<revealed_card2>,...
                    hands = [[], []]
                    chips, bids, opp_hands = clause[1:].split('_')
                    bids = [int(x) for x in bids.split(',')]
                    chips = [int(x) for x in chips.split(',')]
                    hands[active] = [card for card in opp_hands.split(',') if card != '']
                    state = GameState(state.dealer, state.street, state.auction, bids, state.wagers, chips, state.hands, hands, state.community_cards, state)
                elif clause[0] == 'B':
                    # New board cards: B<card1>,<card2>,...
                    state = GameState(state.dealer, state.street, state.auction, state.bids, state.wagers, state.chips,
                                             state.hands, state.opp_hands, clause[1:].split(','), state.parent_state)
                elif clause[0] == 'O':
                    # Showdown reveal — backtrack and inject opponent hand into the tree
                    state = state.parent_state
                    revised_hands = list(state.hands)
                    revised_hands[1-active] = clause[1:].split(',')
                    revised_opp_hands = list(state.opp_hands)
                    revised_opp_hands[active] = clause[1:].split(',')
                    # rebuild history
                    state = GameState(state.dealer, state.street, state.auction, state.bids, state.wagers, state.chips,
                                             revised_hands, revised_opp_hands, state.community_cards, state.parent_state)
                    state = HandResult([0, 0], state.bids, state)
                elif clause[0] == 'D':
                    # Round concluded — record payoff and call on_hand_end
                    assert isinstance(state, HandResult)
                    delta = int(clause[1:])
                    payoffs = [-delta, -delta]
                    payoffs[active] = delta
                    state = HandResult(payoffs, state.bids, state.parent_state)
                    game_info = GameInfo(game_info.bankroll + delta, game_info.time_bank, game_info.round_num)
                    self.pokerbot.on_hand_end(game_info, PokerState(state, active))
                    game_info = GameInfo(game_info.bankroll, game_info.time_bank, game_info.round_num + 1)
                    round_flag = True
                elif clause[0] == 'Q':
                    return
            if round_flag:  # ack the engine
                self.send(ActionCheck())
            else:
                assert active == state.dealer % 2
                action = self.pokerbot.get_move(game_info, PokerState(state, active))
                self.send(action)

def parse_args():
    """Parse command-line arguments for establishing the engine socket connection.

    Expects the port number as a mandatory positional argument and optionally
    accepts a ``--host`` flag.  Designed to be called directly in a bot's
    ``if __name__ == '__main__'`` block alongside ``run_bot()``.

    CLI Usage::

        python3 bot.py [--host HOST] PORT

    Args:
        None: Reads from ``sys.argv`` directly via ``argparse``.

    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            * ``host`` (str) – hostname or IP to connect to (default: ``'localhost'``).
            * ``port`` (int) – TCP port number the engine is listening on.
    """
    parser = argparse.ArgumentParser(prog='python3 player.py')
    parser.add_argument('--host', type=str, default='localhost', help='Host to connect to, defaults to localhost')
    parser.add_argument('port', type=int, help='Port on host to connect to')
    return parser.parse_args()

def run_bot(pokerbot, args):
    """Connect a bot to the engine and run the complete match.

    Validates that ``pokerbot`` inherits from ``BaseBot``, establishes a TCP
    connection to the engine at ``args.host:args.port``, wraps the socket in a
    text-mode file handle, and delegates to ``Runner.run()`` for the remainder
    of the match.  Cleans up the socket file and connection on completion.

    Args:
        pokerbot (BaseBot): A fully initialised bot instance.  Must be a
            subclass of ``BaseBot``; an ``AssertionError`` is raised otherwise.
        args (argparse.Namespace): Parsed CLI arguments containing ``host``
            (str) and ``port`` (int) — typically the return value of
            ``parse_args()``.

    Returns:
        None: Returns after the engine sends the ``'Q'`` quit signal or the
            socket connection is closed.

    Raises:
        AssertionError: If ``pokerbot`` is not an instance of ``BaseBot``.

    Example::

        if __name__ == '__main__':
            run_bot(Player(), parse_args())
    """
    assert isinstance(pokerbot, BaseBot)
    try:
        sock = socket.create_connection((args.host, args.port))
    except OSError:
        print('Could not connect to {}:{}'.format(args.host, args.port))
        return
    socketfile = sock.makefile('rw')
    runner = Runner(pokerbot, socketfile)
    runner.run()
    socketfile.close()
    sock.close()