#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

constexpr int ROWS = 6;
constexpr int COLS = 7;

using Bits = std::bitset<ROWS * COLS>;


enum class Player { NONE = 0, RED = 1, YELLOW = -1 };

enum class NodeType { EXACT, LOWER_BOUND, UPPER_BOUND };

struct TTEntry {
  int32_t score;
  NodeType type;
  int best_move;
};

std::unordered_map<__uint128_t, TTEntry> transposition_table = {};

constexpr int COLUMN_SCORE[] = {100, 200, 700, 1000, 700, 200, 100};
constexpr int COLUMN_ORDER[] = {3, 2, 4, 1, 5, 0, 6};
constexpr Bits BITMASKS[] = {
    15,
    30,
    60,
    120,
    1920,
    3840,
    7680,
    15360,
    245760,
    491520,
    983040,
    1966080,
    31457280,
    62914560,
    125829120,
    251658240,
    4026531840,
    8053063680,
    16106127360,
    32212254720,
    515396075520,
    1030792151040,
    2061584302080,
    4123168604160,
    2113665,
    4227330,
    8454660,
    16909320,
    33818640,
    67637280,
    135274560,
    270549120,
    541098240,
    1082196480,
    2164392960,
    4328785920,
    8657571840,
    17315143680,
    34630287360,
    69260574720,
    138521149440,
    277042298880,
    554084597760,
    1108169195520,
    2216338391040,
    16843009,
    33686018,
    67372036,
    134744072,
    2155905152,
    4311810304,
    8623620608,
    17247241216,
    275955859456,
    551911718912,
    1103823437824,
    2207646875648,
    2130440,
    4260880,
    8521760,
    17043520,
    272696320,
    545392640,
    1090785280,
    2181570560,
    34905128960,
    69810257920,
    139620515840,
    279241031680,
};


class GameState {
  public:
  Bits red;
  Bits yellow;
  Player current_player_;


  public:
  GameState() : current_player_(Player::RED) {}
  GameState(const Bits &red, const Bits &yellow, const Player current_player) :
      red(red), yellow(yellow), current_player_(current_player) {}
  GameState(GameState &&other) noexcept :
      red(other.red), yellow(other.yellow),
      current_player_(other.current_player_) {}
  GameState(const GameState &other) = default;

  // Access board position
  Player operator()(const int row, const int col) const {
    const size_t pos = row * COLS + col;
    if (red.test(pos))
      return Player::RED;
    if (yellow.test(pos))
      return Player::YELLOW;
    return Player::NONE;
  }

  [[nodiscard]] Player current_player() const { return current_player_; }


  // Make a move
  bool make_move(const int col) {
    for (int row = ROWS - 1; row >= 0; --row) {
      const size_t pos = row * COLS + col;
      if (!red.test(pos) && !yellow.test(pos)) {
        if (current_player_ == Player::RED) {
          red.set(pos);
          current_player_ = Player::YELLOW;
        } else {
          yellow.set(pos);
          current_player_ = Player::RED;
        }
        return true;
      }
    }
    return false;
  }

  // Generate legal moves
  [[nodiscard]] std::vector<int> legal_moves() const {
    std::vector<int> moves;
    for (int col : COLUMN_ORDER) {
      if ((*this)(0, col) == Player::NONE) { // Check top row
        moves.push_back(col);
      }
    }
    return moves;
  }

  // Evaluate final position
  [[nodiscard]] int eval_final() const {
    int score = 0;
    for (const auto &mask : BITMASKS) {
      if ((red & mask) == mask)
        score += 1;
      if ((yellow & mask) == mask)
        score -= 1;
    }
    return score;
  }

  // Feature evaluation
  [[nodiscard]] int eval_features(const Bits &mask, const Player turn) const {
    if (mask.count() != 4)
      return 0;

    auto player = (red & mask).count();
    auto opponent = (yellow & mask).count();

    if (turn == Player::YELLOW)
      std::swap(player, opponent);

    if (opponent != 0)
      return 0;

    if (player == 4)
      return 10'000'000; // Feature 1
    if (player == 3)
      return 900'000; // Feature 2.2 & Feature 2.3
    if (player == 2)
      return 50'000; // Feature 3 (approximate)

    return 0;
  }

  // Full evaluation
  [[nodiscard]] int evaluate() const {
    auto moves = legal_moves();
    auto copy = GameState(*this);

    while (moves.size() == 1) {
      copy.make_move(moves[0]);
      moves = copy.legal_moves();
    }

    if (moves.empty()) {
      return copy.eval_final() * 10'000'000;
    }

    int score = 0;
    for (const auto &mask : BITMASKS) {
      score += eval_features(mask, Player::RED);
      score -= eval_features(mask, Player::YELLOW);
    }

    // count number of 3 in a row threats
    for (int row = 0; row < ROWS; ++row) {
      for (int col = 0; col < COLS - 3; ++col) {
        Bits mask;
        for (int i = 0; i < 4; ++i) {
          mask.set(row * COLS + col + i);
        }
        if ((red & mask).count() == 3 && (yellow & mask).count() == 0)
          score += 1'000'000;
      }
    }

    for (int row = 0; row < ROWS; ++row) {
      for (int col = 0; col < COLS; ++col) {
        score += static_cast<int>((*this)(row, col)) * COLUMN_SCORE[col];
      }
    }

    return score;
  }

  // Hashing with symmetry
  [[nodiscard]] __uint128_t hash() const {
    Bits flipped_red, flipped_yellow;

    for (int row = 0; row < ROWS; ++row) {
      for (int col = 0; col < COLS; ++col) {
        const size_t pos = row * COLS + col;
        const size_t flipped_pos = row * COLS + (COLS - 1 - col);

        if (red.test(pos))
          flipped_red.set(flipped_pos);
        if (yellow.test(pos))
          flipped_yellow.set(flipped_pos);
      }
    }

    const auto original =
        (static_cast<__uint128_t>(yellow.to_ullong()) << 64) | red.to_ullong();
    const auto flipped =
        (static_cast<__uint128_t>(flipped_yellow.to_ullong()) << 64) |
        flipped_red.to_ullong();

    return std::min(original, flipped);
    // return original;
  }


  [[nodiscard]] bool is_game_over() const {
    // Check if top row is full
    for (int col = 0; col < COLS; ++col) {
      if (!red.test(col) && !yellow.test(col)) {
        return false;
      }
    }
    return true;
  }
};


std::tuple<int32_t, int> minimax_impl(const GameState &game_state,
                                      const uint32_t depth = 9,
                                      int alpha = INT_MIN, int beta = INT_MAX) {
  // Check transposition table with depth consideration
  const auto game_hash = game_state.hash();
  const auto tt_it = transposition_table.find(game_hash);

  if (tt_it != transposition_table.end()) {
    const auto &[score, type, best_move] = tt_it->second;
    if (type == NodeType::EXACT) {
      return std::make_tuple(score, best_move);
    }
    if (type == NodeType::LOWER_BOUND) {
      alpha = std::max(alpha, score);
    } else if (type == NodeType::UPPER_BOUND) {
      beta = std::min(beta, score);
    }

    if (alpha >= beta) {
      return std::make_tuple(score, best_move);
    }
  }

  // Terminal node check
  if (depth == 0 || game_state.is_game_over()) {
    int32_t score = game_state.evaluate();
    transposition_table[game_hash] = {score, NodeType::EXACT, -1};
    return std::make_tuple(score, -1);
  }

  const bool is_max = game_state.current_player() == Player::RED;
  int best_score = is_max ? INT_MIN : INT_MAX;
  int best_move = -1;

  // Get and order moves (best move from TT first)
  auto legal_moves = game_state.legal_moves();

  // If we have a best move from transposition table, try it first
  if (tt_it != transposition_table.end() && tt_it->second.best_move != -1) {
    const auto best_move_from_tt = tt_it->second.best_move;
    const auto it = std::ranges::find(legal_moves, best_move_from_tt);
    if (it != legal_moves.end()) {
      std::swap(*legal_moves.begin(), *it);
    }
  }

  for (const auto &move : legal_moves) {
    auto next_state = GameState(game_state);
    next_state.make_move(move);

    auto [child_score, _] = minimax_impl(next_state, depth - 1, alpha, beta);

    if (is_max) {
      if (child_score > best_score) {
        best_score = child_score;
        best_move = move;
        alpha = std::max(alpha, best_score);
      }
    } else {
      if (child_score < best_score) {
        best_score = child_score;
        best_move = move;
        beta = std::min(beta, best_score);
      }
    }

    // Alpha-beta pruning
    if (alpha >= beta) {
      break;
    }
  }

  // Store in transposition table
  TTEntry entry{};
  entry.score = best_score;
  entry.best_move = best_move;

  if (best_score <= alpha) {
    entry.type = NodeType::UPPER_BOUND;
  } else if (best_score >= beta) {
    entry.type = NodeType::LOWER_BOUND;
  } else {
    entry.type = NodeType::EXACT;
  }

  transposition_table[game_hash] = entry;

  return std::make_tuple(best_score, best_move);
}

extern "C" {
struct MinimaxRet {
  int score;
  int col;
};

MinimaxRet minimax(const uint64_t red_board, const uint64_t yellow_board,
                   const int player, const int depth) {
  const GameState game_state(red_board, yellow_board,
                             static_cast<Player>(player));
  transposition_table = {};
  transposition_table.reserve(1 << 18); // Reserve space
  auto [score, move] = minimax_impl(game_state, depth);
  return MinimaxRet{score, move};
}
}


int main() {
  // Example usage
  const GameState game;

  auto [score, move] = minimax_impl(game, 7);
  std::cout << "Best move: " << move << ", Score: " << score << std::endl;
  return 0;
}
