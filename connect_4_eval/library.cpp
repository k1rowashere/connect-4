// #include <algorithm>
// #include <cstdint>
// #include <iostream>
// #include <unordered_map>
// #include <vector>
//
// constexpr int ROWS = 6;
// constexpr int COLS = 7;
//
// constexpr int OFFSETS[4][4][2] = {
//     {{0, 0}, {0, 1}, {0, 2}, {0, 3}}, // horizontal
//     {{0, 0}, {1, 0}, {2, 0}, {3, 0}}, // vertical
//     {{0, 0}, {1, 1}, {2, 2}, {3, 3}}, // diagonal down-right
//     {{0, 0}, {1, -1}, {2, -2}, {3, -3}} // diagonal down-left
// };
// constexpr int COLUMN_ORDER[7] = {3, 2, 4, 1, 5, 0, 6};
// constexpr int COLUMN_SCORE[7] = {100, 200, 500, 700, 500, 200, 100};
//
// enum Player { NONE = 0, RED = 1, YELLOW = -1 };
//
// struct Move {
//   int row;
//   int col;
//   bool operator==(const Move &other) const {
//     return row == other.row && col == other.col;
//   }
// };
//
//
// struct GameState {
//   Player board[ROWS][COLS]{};
//   Player current_player;
//
//   GameState(const GameState &other) {
//     for (int r = 0; r < ROWS; ++r)
//       for (int c = 0; c < COLS; ++c)
//         board[r][c] = other.board[r][c];
//     current_player = other.current_player;
//   }
//
//   GameState(GameState &&other) noexcept {
//     for (int r = 0; r < ROWS; ++r)
//       for (int c = 0; c < COLS; ++c)
//         board[r][c] = other.board[r][c];
//     current_player = other.current_player;
//   }
//
//
//   // from int8_t board[ROWS][COLS], int8_t player
//   GameState(const int8_t board[ROWS][COLS], int8_t player) {
//     for (int r = 0; r < ROWS; ++r)
//       for (int c = 0; c < COLS; ++c)
//         this->board[r][c] = static_cast<Player>(board[r][c]);
//     this->current_player = static_cast<Player>(player);
//   }
//
//
//   [[nodiscard]] std::vector<int> legal_moves() const {
//     auto legal_moves = std::vector<int>();
//     legal_moves.reserve(7);
//
//     for (const auto c : COLUMN_ORDER) {
//       if (board[0][c] == NONE) {
//         legal_moves.push_back(c);
//       }
//     }
//
//     return legal_moves;
//   }
//
//   [[nodiscard]] int eval_final() const {
//     int score = 0;
//     for (int r = 0; r < ROWS; ++r) {
//       for (int c = 0; c < COLS; ++c) {
//         const auto cell = board[r][c];
//         if (cell == NONE)
//           continue;
//
//         for (const auto &offset : OFFSETS) {
//           int bounds[2][2] = {{r + offset[0][0], c + offset[0][1]},
//                               {r + offset[3][0], c + offset[3][1]}};
//
//           auto within_bounds = [&](auto &o) {
//             return o[0] >= 0 && o[0] < ROWS && o[1] >= 0 && o[1] < COLS;
//           };
//
//           if (std::ranges::all_of(bounds, within_bounds)) {
//             score += cell * std::ranges::all_of(offset, [&](const auto &o) {
//                        return board[r + o[0]][c + o[1]] == cell;
//                      });
//           }
//         }
//       }
//     }
//     return score;
//   }
//
//   int eval_features(const std::vector<int> &legal_moves, int offset[4][2],
//                     const Player player) const {
//     // Check bounds
//     for (int i = 0; i < 4; ++i) {
//       const int r = offset[i][0];
//       const int c = offset[i][1];
//       if (r < 0 || r >= ROWS || c < 0 || c >= COLS) {
//         return 0;
//       }
//     }
//
//     // Get quad values
//     int8_t quad[4];
//     int player_count = 0;
//     int opponent_count = 0;
//     const Player opponent = (player == RED) ? YELLOW : RED;
//
//     for (int i = 0; i < 4; ++i) {
//       quad[i] = board[offset[i][0]][offset[i][1]];
//       if (quad[i] == player)
//         player_count++;
//       if (quad[i] == opponent)
//         opponent_count++;
//     }
//
//     if (opponent_count > 0 || player_count == 0) {
//       return 0;
//     }
//
//     if (player_count == 4) {
//       return 10'000'000;
//     }
//     if (player_count == 3) {
//       // Calculate adjacent square
//       const int adj_r = 2 * offset[0][0] - offset[1][0];
//       const int adj_c = 2 * offset[0][1] - offset[1][1];
//       const Move adj_move = {adj_r, adj_c};
//       const Move last_move = {offset[3][0], offset[3][1]};
//
//       const bool adj_legal = std::ranges::contains(legal_moves, adj_move.col)
//       &&
//           board[adj_r][adj_c] == NONE;
//       const bool last_legal =
//           std::ranges::contains(legal_moves, last_move.col) &&
//           board[last_move.row][last_move.col] == NONE;
//
//       if (adj_legal && last_legal)
//         return 10'000'000;
//       return 900'000;
//     }
//     if ((quad[0] == quad[1] && quad[0] == player) ||
//         (quad[1] == quad[2] && quad[1] == player) ||
//         (quad[2] == quad[3] && quad[2] == player)) {
//       return 20'000;
//     }
//
//     return 0;
//   }
//
//   [[nodiscard]] int evaluate() const {
//     auto legal_moves_vec = legal_moves();
//     int score = 0;
//
//     // Check for immediate win/loss (1 or 0 legal moves left)
//     auto game_copy = GameState(*this);
//     while (legal_moves_vec.size() == 1) {
//       game_copy.make_move(legal_moves_vec.back());
//       legal_moves_vec = game_copy.legal_moves();
//     }
//
//     if (legal_moves_vec.empty()) {
//       return game_copy.eval_final() * 10'000'000;
//     }
//
//     // Evaluate features
//     for (int r = 0; r < ROWS; ++r)
//       for (int c = 0; c < COLS; ++c) {
//         const auto cell = board[r][c];
//
//         // Add column score
//         score += cell * COLUMN_SCORE[c];
//
//         // Check all offset patterns
//         for (const auto &o : OFFSETS) {
//           int current_offset[4][2];
//           for (int i = 0; i < 4; ++i) {
//             current_offset[i][0] = r + o[i][0];
//             current_offset[i][1] = c + o[i][1];
//           }
//
//           score += eval_features(legal_moves_vec, current_offset, RED);
//           score -= eval_features(legal_moves_vec, current_offset, YELLOW);
//         }
//       }
//
//
//     return score;
//   }
//
//   bool make_move(const int move_column) {
//     for (int r = ROWS - 1; r >= 0; --r)
//       if (board[r][move_column] == NONE) {
//         board[r][move_column] = current_player;
//         current_player = static_cast<Player>(-current_player);
//         return true;
//       }
//     return false;
//   }
//
//   [[nodiscard]] __uint128_t hash() const {
//     uint64_t yellow = 0;
//     uint64_t red = 0;
//
//     uint64_t flipped_yellow = 0;
//     uint64_t flipped_red = 0;
//
//     for (int r = 0; r < ROWS; ++r) {
//       for (int c = 0; c < COLS; ++c) {
//         if (board[r][c] == YELLOW) {
//           yellow |= (1ULL << (r * COLS + c));
//           flipped_yellow |= (1ULL << (r * COLS + (COLS - 1 - c)));
//         } else if (board[r][c] == RED) {
//           red |= (1ULL << (r * COLS + c));
//           flipped_red |= (1ULL << (r * COLS + (COLS - 1 - c)));
//         }
//       }
//     }
//
//     if (flipped_yellow < yellow ||
//         (flipped_yellow == yellow && flipped_red < red)) {
//       yellow = flipped_yellow;
//       red = flipped_red;
//     }
//
//     return (static_cast<__uint128_t>(yellow) << 64) | red;
//   }
//
//   [[nodiscard]] bool is_game_over() const { return legal_moves().empty(); }
// };
//
//
//
//
//
//
//
//
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

std::unordered_map<__uint128_t, int32_t> transposition_table = {};

// Initialize bitmasks (should be done once at startup)
// const std::vector<Bits> BITMASKS = []() {
//     std::vector<Bits> masks;
//
//     // Horizontal
//     for (int row = 0; row < ROWS; ++row) {
//         for (int col = 0; col < COLS - 3; ++col) {
//             Bits mask;
//             for (int i = 0; i < 4; ++i) {
//                 mask.set(row * COLS + col + i);
//             }
//             masks.push_back(mask);
//         }
//     }
//
//     // Vertical
//     for (int row = 0; row < ROWS - 3; ++row) {
//         for (int col = 0; col < COLS; ++col) {
//             Bits mask;
//             for (int i = 0; i < 4; ++i) {
//                 mask.set((row + i) * COLS + col);
//             }
//             masks.push_back(mask);
//         }
//     }
//
//     // Diagonal down-right
//     for (int row = 0; row < ROWS - 3; ++row) {
//         for (int col = 0; col < COLS - 3; ++col) {
//             Bits mask;
//             for (int i = 0; i < 4; ++i) {
//                 mask.set((row + i) * COLS + col + i);
//             }
//             masks.push_back(mask);
//         }
//     }
//
//     // Diagonal down-left
//     for (int row = 0; row < ROWS - 3; ++row) {
//         for (int col = 3; col < COLS; ++col) {
//             Bits mask;
//             for (int i = 0; i < 4; ++i) {
//                 mask.set((row + i) * COLS + col - i);
//             }
//             masks.push_back(mask);
//         }
//     }
//
//     return masks;
// }();
//

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


std::bitset<4> collect_bits(const Bits num, const Bits mask) {
  std::bitset<4> result;
  const auto first = mask._Find_first();
  if (first == ROWS * COLS)
    return 0;
  result[0] = num.test(first);

  size_t bit_pos = 0;
  size_t result_pos = 1;

  while (bit_pos < ROWS * COLS) {
    bit_pos = mask._Find_next(bit_pos);
    if (bit_pos == ROWS * COLS)
      break;

    result[result_pos++] = num.test(bit_pos);
  }

  return result;
}

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
  [[nodiscard]] int eval_features(const Bits &mask, Player turn) const {
    if (mask.count() != 4)
      return 0;

    std::bitset<4> player = collect_bits(red, mask);
    std::bitset<4> opponent = collect_bits(yellow, mask);

    if (turn == Player::YELLOW)
      std::swap(player, opponent);

    if (opponent != 0)
      return 0;

    if (player == 0b1111)
      return 10'000'000; // Feature 1
    if (player == 0b0111 || player == 0b1011 || player == 0b1101 ||
        player == 0b1110)
      return 900'000; // Feature 2.2 & Feature 2.3
    if (player == 0b1100 || player == 0b0110 || player == 0b0011)
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
  // Check transposition table
  const auto game_hash = game_state.hash();
  const auto tt_it = transposition_table.find(game_hash);
  if (tt_it != transposition_table.end()) {
    return std::make_tuple(tt_it->second, -1);
  }

  // Terminal node check
  if (depth <= 0 || game_state.is_game_over()) {
    int32_t score = game_state.evaluate();
    transposition_table[game_hash] = score;
    return std::make_tuple(score, -1);
  }

  const bool is_max = game_state.current_player() == Player::RED;
  int best_score = is_max ? INT_MIN : INT_MAX;
  int best_move = -1;

  const auto legal_moves = game_state.legal_moves();

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
  transposition_table[game_hash] = best_score;

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
