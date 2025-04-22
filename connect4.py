import copy
import ctypes
import time
from enum import Enum
from typing import Generator, Tuple

import numpy as np
from graphviz import Digraph

ENABLE_ALPHA_BETA = True
ENABLE_EXPECTIMINIMAX = False


class MinimaxRet(ctypes.Structure):
    _fields_ = [
        ("score", ctypes.c_int),
        ("col", ctypes.c_int),
    ]


lib = ctypes.CDLL(r"./libconnect4_eval.dll")
lib.minimax.argtypes = [
    ctypes.c_uint64,
    ctypes.c_uint64,
    ctypes.c_int,
    ctypes.c_int,
]
lib.minimax.restype = MinimaxRet

ROWS, COLUMNS = 6, 7


def generate_bitmask(row: int, col: int) -> np.ndarray:
    """
    Generate bitmasks for all possible winning positions in a Connect-4 game.
    :param row: Number of rows
    :param col: Number of columns
    :return: Array of bitmasks
    """
    col = int(col)
    row = int(row)
    v = 1 | (1 << col) | (1 << (2 * col)) | (1 << (3 * col))
    dr = 1 | (1 << (col + 1)) | (1 << (2 * col + 2)) | (1 << (3 * col + 3))
    dl = 1 | (1 << (col - 1)) | (1 << (2 * col - 2)) | (1 << (3 * col - 3))
    horizontal = [0b1111 << (i * col + j) for i in range(row) for j in range(col - 3)]
    vertical = [v << (i * col + j) for i in range(row - 3) for j in range(col)]
    diagonal_dr = [dr << (i * col + j) for i in range(row - 3) for j in range(col - 3)]
    diagonal_dl = [dl << (i * col + j) for i in range(row - 3) for j in range(3, col)]
    return np.array([*horizontal, *vertical, *diagonal_dr, *diagonal_dl])


BITMASKS = generate_bitmask(ROWS, COLUMNS)
COLUMN_SCORE = [100, 200, 500, 800, 500, 200, 100]
TRANSPOSITION_TABLE = {}


class Player(Enum):
    NONE = 0
    RED = 1
    YELLOW = -1


class GameState:
    def __init__(self):
        assert ROWS <= 8 and COLUMNS <= 8, "Board too large for 64-bit representation"

        # Bitboard representation (2x 64-bit integers)
        self.red = 0  # Bitmask for red pieces (Player.RED)
        self.yellow = 0  # Bitmask for yellow pieces (Player.YELLOW)
        self.turn = Player.RED

    def __getitem__(self, key: Tuple[int, int]) -> Player:
        row, col = key
        mask = 1 << (row * COLUMNS + col)

        if self.red & mask:
            return Player.RED
        elif self.yellow & mask:
            return Player.YELLOW
        else:
            return Player.NONE

    def __hash__(self):
        # to respect the symmetry of the board,
        # the hash of the flipped board is the same as the original
        # where the original is the smaller of the two

        flipped_yellow = 0
        flipped_red = 0
        mask = (1 << COLUMNS) - 1  # Mask for one row
        tmp_yellow = self.yellow
        tmp_red = self.red

        # flip each row
        for row in range(ROWS):
            yellow_bits = bin(tmp_yellow & mask)[2:].zfill(COLUMNS)
            red_bits = bin(tmp_red & mask)[2:].zfill(COLUMNS)

            reversed_yellow = int(yellow_bits[::-1], 2)
            reversed_red = int(red_bits[::-1], 2)

            flipped_yellow |= reversed_yellow << (row * COLUMNS)
            flipped_red |= reversed_red << (row * COLUMNS)

            tmp_yellow >>= COLUMNS
            tmp_red >>= COLUMNS

        original_hash = (self.red << 64) | self.yellow
        flipped_hash = (flipped_red << 64) | flipped_yellow

        return min(original_hash, flipped_hash)

    def __repr__(self):
        board_str = ""
        for row in range(ROWS):
            board_str += "|"
            for col in range(COLUMNS):
                player = self[row, col]
                if player == Player.RED:
                    board_str += "x|"
                elif player == Player.YELLOW:
                    board_str += "o|"
                else:
                    board_str += " |"
            board_str += "\n"
        return board_str

    def print_colored(self) -> str:
        board_str = (str(self)
                     .replace('x', '\033[91m●\033[0m')
                     .replace('o', '\033[93m●\033[0m'))
        return board_str

    def is_game_over(self):
        mask = 2 ** COLUMNS - 1
        return (self.yellow | self.red) & mask == mask

    def make_move(self, move_col: int) -> None:
        if self[0, move_col] != Player.NONE:
            raise ValueError(f"Column {move_col} is full")
        # Find the lowest empty row in the column
        row = ROWS - 1
        while row >= 0 and self[row, move_col] != Player.NONE:
            row -= 1

        if self.turn == Player.RED:
            self.red |= 1 << (row * COLUMNS + move_col)
            self.turn = Player.YELLOW
        else:
            self.yellow |= 1 << (row * COLUMNS + move_col)
            self.turn = Player.RED

    @staticmethod
    def column_order():
        n = COLUMNS // 2
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
        for bitmask in BITMASKS:
            if self.red & bitmask == bitmask:
                score += 1
            if self.yellow & bitmask == bitmask:
                score -= 1
        return score

    def eval_features(self, bitmask: int, turn: Player) -> int:
        # mask within the board there are 4 ones
        if bitmask.bit_count() != 4:
            return 0

        # collect bits
        player = (self.red & bitmask).bit_count()
        opponent = (self.yellow & bitmask).bit_count()

        # the left adjacent piece (to check for 3 in a row, with 2 empty spaces)
        # adjacent = bitmask.first_bit_set() - 1

        if turn == Player.YELLOW:
            player, opponent = opponent, player

        if opponent != 0:
            return 0  # this will never lead to a connect-4

        if player == 4:
            return 15_000_000  # feature 1
        elif player == 3:
            return 900_000  # feature 2.2 & feature 2.3
        elif player == 2:
            return 50_000  # feature 3 (approximate)

        return 0

    def evaluate(self) -> int:
        """
        https://researchgate.net/publication/331552609_Research_on_Different_Heuristics_for_Minimax_Algorithm_Insight_from_Connect-4_Game
        """

        # if top row as a single column open, make that move
        # if top row has no open columns, return score
        legal_moves = list(self.legal_moves())
        game_copy = copy.deepcopy(self)
        while len(legal_moves) == 1:
            game_copy.make_move(legal_moves[0])
            legal_moves = list(game_copy.legal_moves())

        if len(legal_moves) == 0:
            return game_copy.evaluate_final() * 15_000_000

        score = 0
        for bitmask in BITMASKS:
            score += self.eval_features(bitmask, Player.RED)
            score -= self.eval_features(bitmask, Player.YELLOW)

        for r in range(ROWS):
            for c in range(COLUMNS):
                score += self[r, c].value * COLUMN_SCORE[c]  # feature 4

        return score


class TreeNode:
    def __init__(self, board: GameState, score=None, move=None):
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
    dot.node(str(id(node)),
             label=node_label,
             shape='box',
             style='filled,rounded',
             fillcolor='#ffcccc' if node.board.turn == Player.RED else '#ffffcc',
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
        return score, None, TreeNode(game_state, score)

    if depth == 0 or game_state.is_game_over():
        score = game_state.evaluate()
        TRANSPOSITION_TABLE[game_hash] = score
        return score, None, TreeNode(game_state, score)

    is_max = game_state.turn == Player.RED
    best_score = -float('inf') if is_max else float('inf')
    best_move = None
    scores = [-1] * 7
    legal_moves = list(game_state.legal_moves())
    tree = TreeNode(game_state)

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
                best_score = 0.4 * scores[l] + 0.6 * scores[best_move]
                best_move = rng.choice(moves, p=[0.6, 0.4, 0.0])
            case (None, r):
                best_score = 0.4 * scores[r] + 0.6 * scores[best_move]
                best_move = rng.choice(moves, p=[0.6, 0.0, 0.4])
            case (l, r):
                best_score = 0.2 * scores[l] + 0.2 * scores[r] + 0.6 * scores[best_move]
                best_move = rng.choice(moves, p=[0.6, 0.2, 0.2])
        best_move = int(best_move)

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
    ret = lib.minimax(game_state.red, game_state.yellow, game_state.turn.value, depth)
    return ret.score, ret.col


def main():
    game = GameState()

    while not game.is_game_over():
        TRANSPOSITION_TABLE.clear()
        start = time.time()
        if False:
            score, move = minimax_cpp(game, 11)
        else:
            score, move, tree = minimax(game, 4)
        print(f"Bot {game.turn}: Move: {move} ({time.time() - start:.2f}s)")
        print(f"Score: {score}")
        build_graph(tree).render('game_tree')

        game.make_move(move)
        print(game.print_colored())
        # print(game)

        break
        if game.is_game_over():
            break

    print("Game over")
    print(game.evaluate_final())


def play(game, engine, depth):
    TRANSPOSITION_TABLE.clear()
    global ENABLE_ALPHA_BETA
    global ENABLE_EXPECTIMINIMAX

    match engine:
        case "human":
            raise NotImplementedError("Human player not implemented")
        case "minimax_python":
            ENABLE_EXPECTIMINIMAX = False
            ENABLE_ALPHA_BETA = False
            return minimax(game, depth)
        case "minimax_ab_python":
            ENABLE_EXPECTIMINIMAX = False
            ENABLE_ALPHA_BETA = True
            return minimax(game, depth)
        case "expectiminimax":
            ENABLE_EXPECTIMINIMAX = True
            ENABLE_ALPHA_BETA = False
            return minimax(game, depth)
        case "minimax_cpp":
            score, move = minimax_cpp(game, depth)
            return score, move, None
        case _:
            raise ValueError(f"Unknown engine: {engine}")


if __name__ == "__main__":
    main()
