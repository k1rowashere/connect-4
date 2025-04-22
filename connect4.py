import copy
import time
from enum import Enum
from typing import Generator, Tuple
import ctypes

from graphviz import Digraph
from numpy.ctypeslib import ndpointer

import numpy as np

ENABLE_ALPHA_BETA = True
ENABLE_EXPECTIMINIMAX = False
ENABLE_CPP_MINIMAX = True

TRANSPOSITION_TABLE = {}

OFFSETS = np.array([
    [(0, 0), (0, 1), (0, 2), (0, 3)],  # horizontal
    [(0, 0), (1, 0), (2, 0), (3, 0)],  # vertical
    [(0, 0), (1, 1), (2, 2), (3, 3)],  # diagonal down-right
    [(0, 0), (1, -1), (2, -2), (3, -3)],  # diagonal down-left
], dtype=np.int8)

COLUMN_SCORE = [100, 200, 500, 800, 500, 200, 100]


class Player(Enum):
    NONE = 0
    RED = 1
    YELLOW = -1


class TreeNode:
    def __init__(self, board=0, score=None, move=None):
        self.board = board
        self.move = move
        self.score = score
        self.children = []


def build_graph(node, dot=None, parent_label=None, depth=0, max_depth=5):
    if dot is None:  # initialize graph
        dot = Digraph(format='svg')
        dot.attr(rankdir='TB')  # Top to bottom layout
        dot.attr('node', fontsize='10', fontname='Arial')

    # Calculate node size based on depth (smaller as depth increases)
    depth_factor = max(0.2, 1.0 - (depth / (max_depth * 1.5)))  # scales from 1.0 to ~0.2
    node_size = str(0.5 + depth_factor * 1.5)  # between 0.5 and 2.0 inches

    move_char = "ROOT" if node.move is None else chr(node.move + 65)
    node_label = f"{move_char}\n{node.score}"

    # Color based on player
    red = bin(node.board & (2 ** 64 - 1)).count('1')
    yellow = bin(node.board >> 64).count('1')
    dot.node(str(id(node)),
             label=node_label,
             shape='box',
             style='filled,rounded',
             fillcolor='#ffcccc' if yellow >= red else '#ffffcc',
             color='gray',
             fontsize=str(24 + (1 - depth_factor) * 6),  # Shrink font with depth
             width=node_size,
             height=node_size,
             penwidth=str(1 + depth_factor * 2))  # Thicker borders for top nodes

    # Connect to parent if exists
    if parent_label is not None:
        edge_penwidth = max(0.5, 2.0 - depth / 3)  # Thicker lines for higher levels
        dot.edge(parent_label, str(id(node)),
                 penwidth=str(edge_penwidth),
                 color='#666666')

    # Recursively build graph for children (with depth limit)
    if depth < max_depth:
        for child in sorted(node.children, key=lambda x: abs(x.score), reverse=True):
            build_graph(child, dot, str(id(node)), depth + 1, max_depth)

    # Add legend for top level
    if depth == 0:
        with dot.subgraph(name='cluster_legend') as legend:
            legend.attr(label='Legend', style='dashed', fontsize='10')
            legend.node('max', 'MAX\nScore', shape='box', style='filled', fillcolor='#ffcccc')
            legend.node('min', 'MIN\nScore', shape='box', style='filled', fillcolor='#ffffcc')
            legend.node('neutral', 'Neutral', shape='box', style='filled', fillcolor='#e6e6e6')
            legend.attr(rank='same')

    return dot


class Move:
    def __init__(self, row, column):
        self.row = row
        self.column = column

    def __repr__(self):
        return f"M({self.row}, {self.column})"

    # spread
    def __iter__(self):
        yield self.row
        yield self.column


class GameState:
    def __init__(self, dim: Tuple[int, int] = (6, 7)):
        self.ROWS, self.COLUMNS = dim
        self.board = np.zeros((self.ROWS, self.COLUMNS), dtype=np.int8)
        self.turn = Player.RED  # player about to play

    def __getitem__(self, key) -> np.ndarray[Player] | Player:
        arr = self.board[key]
        if isinstance(arr, np.ndarray):
            return np.vectorize(lambda x: Player(x))(arr)
        return Player(arr)

    def __setitem__(self, key, value: Player):
        self.board[key] = value.value

    def __repr__(self) -> str:
        board_str = ""
        for row in range(self.ROWS):
            board_str += "|"
            for col in range(self.COLUMNS):
                board_str += 'x' if self[row, col] == Player.RED else ''
                board_str += 'o' if self[row, col] == Player.YELLOW else ''
                board_str += ' ' if self[row, col] == Player.NONE else ''
                board_str += '|'
            board_str += "\n"
        return board_str

    def __hash__(self):
        yellow = int.from_bytes(np.packbits(self.board == Player.YELLOW.value))
        red = int.from_bytes(np.packbits(self.board == Player.RED.value))

        # canonical form, yellow is the smaller number
        # hash of flipped board == hash of original board
        # to ensure symmetrical positions are not duplicated
        flipped = np.flip(self.board, axis=1)
        flipped_yellow = int.from_bytes(np.packbits(flipped == Player.YELLOW.value))
        flipped_red = int.from_bytes(np.packbits(flipped == Player.RED.value))

        if flipped_yellow < yellow \
                or flipped_yellow == yellow \
                and flipped_red < red:
            yellow = flipped_yellow
            red = flipped_red

        return yellow << 64 | red

    def print_colored(self) -> str:
        board_str = (str(self)
                     .replace('x', '\033[91m●\033[0m')
                     .replace('o', '\033[93m●\033[0m'))
        return board_str

    def is_game_over(self):
        return np.all(self[0] != Player.NONE)

    def make_move(self, move_col: int) -> None:
        if self[0, move_col] != Player.NONE:
            raise ValueError(f"Column {move_col} is full")
        row = np.argwhere(self[:, move_col] == Player.NONE)[-1, 0]
        self[row, move_col] = self.turn
        self.turn = Player.YELLOW if self.turn == Player.RED else Player.RED

    def column_order(self):
        n = self.COLUMNS // 2
        yield n
        i = 1
        while i <= n:
            yield n + i
            yield n - i
            i += 1

    def legal_moves(self) -> Generator[int, None, None]:
        """
        valid moves, with move ordering optimisation (center moves first)
        :return: iterator for moves
        """

        for c in self.column_order():
            if self[0, c] == Player.NONE:
                yield c

    def evaluate_final(self) -> int:
        score = 0
        for r in range(self.ROWS):
            for c in range(self.COLUMNS):
                for offset in OFFSETS:
                    o = np.add([r, c], offset)
                    if np.any(o < 0) or np.any(o >= [self.ROWS, self.COLUMNS]):
                        continue
                    if np.all(self[o[:, 0], o[:, 1]] == self[r, c]):
                        score += self[r, c].value
        return score

    def eval_features(self, offset, player: Player) -> int:
        if np.any(offset < 0) or np.any(offset >= [self.ROWS, self.COLUMNS]):
            return 0
        quad = self[offset[:, 0], offset[:, 1]]
        offset_1 = 2 * offset[0] - offset[1]  # adjacent square to quad

        player_1_count = np.count_nonzero(quad == player)
        player_2_count = np.count_nonzero(quad == Player.YELLOW if player == Player.RED else Player.RED)
        if player_2_count > 0 or player_1_count == 0:
            return 0  # this will never lead to a connect-4

        if player_1_count == 4:
            return 10_000_000  # feature 1
        elif player_1_count == 3:
            try:
                if self[offset_1[0], offset_1[1]] == Player.NONE and \
                        self[offset_1[0], offset_1[1] - 1] == Player.NONE \
                        and self[offset[3, 0], offset[3, 1]] == Player.NONE \
                        and self[offset[3, 0], offset[3, 1] - 1] == Player.NONE:
                    return 10_000_000  # feature 2.1
            except IndexError:
                pass
            return 900_000
        elif quad[0] == quad[1] == player \
                or quad[1] == quad[2] == player \
                or quad[2] == quad[3] == player:
            return 50_000  # feature 3 (approximate)

        return 0

    def evaluate(self) -> int:
        """
        https://researchgate.net/publication/331552609_Research_on_Different_Heuristics_for_Minimax_Algorithm_Insight_from_Connect-4_Game
        """

        # if top row as a single column open, make that move
        # if top row has no open columns, return score
        open_cols = np.argwhere(self[0] == Player.NONE)
        game_copy = copy.deepcopy(self)
        while open_cols.size == 1:
            # noinspection PyTypeChecker
            game_copy.make_move(open_cols[0])
            open_cols = np.argwhere(game_copy[0] == Player.NONE)
        if open_cols.size == 0:
            return game_copy.evaluate_final() * 10_000_000

        score = 0
        for r in range(self.ROWS):
            for c in range(self.COLUMNS):
                for offset in OFFSETS:
                    score += self.eval_features(np.add([r, c], offset), Player.RED)
                    score -= self.eval_features(np.add([r, c], offset), Player.YELLOW)
                score += self[r, c].value * COLUMN_SCORE[c]  # feature 4

        return score


def minimax(game_state: GameState,
            depth=9,
            alpha=-float('inf'),
            beta=float('inf')) \
        -> Tuple[int, int | None, TreeNode | None]:
    global ENABLE_ALPHA_BETA
    global ENABLE_EXPECTIMINIMAX

    game_hash = game_state.__hash__()
    if game_hash in TRANSPOSITION_TABLE:
        score = TRANSPOSITION_TABLE[game_hash]
        return score, None, TreeNode(game_hash, score)

    if depth == 0 or game_state.is_game_over():
        score = game_state.evaluate()
        TRANSPOSITION_TABLE[game_hash] = score
        return score, None, TreeNode(game_hash, score)

    is_max = game_state.turn == Player.RED
    best_score = -float('inf') if is_max else float('inf')
    best_move = None
    scores = [-1] * 7
    legal_moves = list(game_state.legal_moves())
    tree = TreeNode(game_hash)

    for move in legal_moves:
        next_game_state = copy.deepcopy(game_state)
        next_game_state.make_move(move)
        child_score, _, next_node = minimax(next_game_state, depth - 1, alpha, beta)
        next_node.move = move
        tree.children.append(next_node)

        if is_max:
            if child_score > best_score:
                best_score = child_score
                best_move = move
                alpha = max(alpha, best_score)
        else:
            if child_score < best_score:
                best_score = child_score
                best_move = move
                beta = min(beta, best_score)

        scores[move] = child_score

        # Alpha-beta pruning
        if ENABLE_ALPHA_BETA and not ENABLE_EXPECTIMINIMAX and alpha >= beta:
            break

    if ENABLE_EXPECTIMINIMAX:
        moves = [best_move, best_move - 1, best_move + 1]
        if moves[1] not in legal_moves:
            moves[1] = None
        if moves[2] not in legal_moves:
            moves[2] = None

        rng = np.random.default_rng()
        match (moves[1], moves[2]):
            case (None, None):
                best_move = moves[0]
                best_score = scores[best_move]
            case (l, None):
                best_score = scores[l] + 0.4 * scores[best_move]
                best_move = rng.choice(moves, p=[0.6, 0.4, 0.0])
            case (None, r):
                best_score = 0.4 * scores[r] + 0.6 * scores[best_move]
                best_move = rng.choice(moves, p=[0.6, 0.0, 0.4])
            case (l, r):
                best_score = 0.2 * scores[l] + 0.2 * scores[r] + 0.6 * scores[best_move]
                best_move = rng.choice(moves, p=[0.6, 0.2, 0.2])

    tree.score = best_score

    TRANSPOSITION_TABLE[game_hash] = best_score
    return best_score, best_move, tree


def minimax_cpp(game_state: GameState, depth=9) -> Tuple[int, int]:
    """
    Call the C++ library to evaluate the board
    :param game_state: current game state
    :param depth: depth to search
    :return: score and move
    """
    ret = lib.minimax(game_state.board, game_state.turn.value, depth)
    return ret.score, ret.col


if __name__ == "__main__":
    # load the C++ library

    if ENABLE_CPP_MINIMAX:
        class MinimaxRet(ctypes.Structure):
            _fields_ = [
                ("score", ctypes.c_int),
                ("col", ctypes.c_int),
            ]


        lib = ctypes.CDLL(r"./libconnect4_eval.dll")
        lib.minimax.argtypes = [
            ndpointer(dtype=np.int8, ndim=2, flags='C_CONTIGUOUS'),
            ctypes.c_int8,
            ctypes.c_int,
        ]
        lib.minimax.restype = MinimaxRet

    game = GameState()

    while not game.is_game_over():
        TRANSPOSITION_TABLE.clear()
        start = time.time()
        score, move, tree = minimax(game, 3)
        # build_graph(tree).render('game_tree')
        print(f"Bot {game.turn}: Move: {move} ({time.time() - start:.2f}s)")

        game.make_move(move)
        print(game.print_colored())
        # print(game)

        # break
        if game.is_game_over():
            break

    print("Game over")
    print(game.evaluate_final())
