#include <algorithm>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

constexpr int ROWS = 6;
constexpr int COLS = 7;

constexpr int OFFSETS[4][4][2] = {
    {{0, 0}, {0, 1}, {0, 2}, {0, 3}}, // horizontal
    {{0, 0}, {1, 0}, {2, 0}, {3, 0}}, // vertical
    {{0, 0}, {1, 1}, {2, 2}, {3, 3}}, // diagonal down-right
    {{0, 0}, {1, -1}, {2, -2}, {3, -3}} // diagonal down-left
};
constexpr int COLUMN_ORDER[7] = {3, 2, 4, 1, 5, 0, 6};
constexpr int COLUMN_SCORE[7] = {100, 200, 500, 700, 500, 200, 100};

enum Player { NONE = 0, RED = 1, YELLOW = -1 };

struct Move {
  int row;
  int col;
  bool operator==(const Move &other) const {
    return row == other.row && col == other.col;
  }
};

struct GameState {
  Player board[ROWS][COLS]{};
  Player current_player;

  GameState(const GameState &other) {
    for (int r = 0; r < ROWS; ++r)
      for (int c = 0; c < COLS; ++c)
        board[r][c] = other.board[r][c];
    current_player = other.current_player;
  }

  GameState(GameState &&other) noexcept {
    for (int r = 0; r < ROWS; ++r)
      for (int c = 0; c < COLS; ++c)
        board[r][c] = other.board[r][c];
    current_player = other.current_player;
  }


  // from int8_t board[ROWS][COLS], int8_t player
  GameState(const int8_t board[ROWS][COLS], int8_t player) {
    for (int r = 0; r < ROWS; ++r)
      for (int c = 0; c < COLS; ++c)
        this->board[r][c] = static_cast<Player>(board[r][c]);
    this->current_player = static_cast<Player>(player);
  }


  [[nodiscard]] std::vector<int> legal_moves() const {
    auto legal_moves = std::vector<int>();
    legal_moves.reserve(7);

    for (const auto c : COLUMN_ORDER) {
      if (board[0][c] == NONE) {
        legal_moves.push_back(c);
      }
    }

    return legal_moves;
  }

  [[nodiscard]] int eval_final() const {
    int score = 0;
    for (int r = 0; r < ROWS; ++r) {
      for (int c = 0; c < COLS; ++c) {
        const auto cell = board[r][c];
        if (cell == NONE)
          continue;

        for (const auto &offset : OFFSETS) {
          int bounds[2][2] = {{r + offset[0][0], c + offset[0][1]},
                              {r + offset[3][0], c + offset[3][1]}};

          auto within_bounds = [&](auto &o) {
            return o[0] >= 0 && o[0] < ROWS && o[1] >= 0 && o[1] < COLS;
          };

          if (std::ranges::all_of(bounds, within_bounds)) {
            score += cell * std::ranges::all_of(offset, [&](const auto &o) {
                       return board[r + o[0]][c + o[1]] == cell;
                     });
          }
        }
      }
    }
    return score;
  }

  int eval_features(const std::vector<int> &legal_moves, int offset[4][2],
                    const Player player) const {
    // Check bounds
    for (int i = 0; i < 4; ++i) {
      const int r = offset[i][0];
      const int c = offset[i][1];
      if (r < 0 || r >= ROWS || c < 0 || c >= COLS) {
        return 0;
      }
    }

    // Get quad values
    int8_t quad[4];
    int player_count = 0;
    int opponent_count = 0;
    const Player opponent = (player == RED) ? YELLOW : RED;

    for (int i = 0; i < 4; ++i) {
      quad[i] = board[offset[i][0]][offset[i][1]];
      if (quad[i] == player)
        player_count++;
      if (quad[i] == opponent)
        opponent_count++;
    }

    if (opponent_count > 0 || player_count == 0) {
      return 0;
    }

    if (player_count == 4) {
      return 10'000'000;
    }
    if (player_count == 3) {
      // Calculate adjacent square
      const int adj_r = 2 * offset[0][0] - offset[1][0];
      const int adj_c = 2 * offset[0][1] - offset[1][1];
      const Move adj_move = {adj_r, adj_c};
      const Move last_move = {offset[3][0], offset[3][1]};

      const bool adj_legal = std::ranges::contains(legal_moves, adj_move.col) &&
          board[adj_r][adj_c] == NONE;
      const bool last_legal =
          std::ranges::contains(legal_moves, last_move.col) &&
          board[last_move.row][last_move.col] == NONE;

      if (adj_legal && last_legal)
        return 10'000'000;
      return 900'000;
    }
    if ((quad[0] == quad[1] && quad[0] == player) ||
        (quad[1] == quad[2] && quad[1] == player) ||
        (quad[2] == quad[3] && quad[2] == player)) {
      return 20'000;
    }

    return 0;
  }

  [[nodiscard]] int evaluate() const {
    auto legal_moves_vec = legal_moves();
    int score = 0;

    // Check for immediate win/loss (1 or 0 legal moves left)
    auto game_copy = GameState(*this);
    while (legal_moves_vec.size() == 1) {
      game_copy.make_move(legal_moves_vec.back());
      legal_moves_vec = game_copy.legal_moves();
    }

    if (legal_moves_vec.empty()) {
      return game_copy.eval_final() * 10'000'000;
    }

    // Evaluate features
    for (int r = 0; r < ROWS; ++r)
      for (int c = 0; c < COLS; ++c) {
        const auto cell = board[r][c];

        // Add column score
        score += cell * COLUMN_SCORE[c];

        // Check all offset patterns
        for (const auto &o : OFFSETS) {
          int current_offset[4][2];
          for (int i = 0; i < 4; ++i) {
            current_offset[i][0] = r + o[i][0];
            current_offset[i][1] = c + o[i][1];
          }

          score += eval_features(legal_moves_vec, current_offset, RED);
          score -= eval_features(legal_moves_vec, current_offset, YELLOW);
        }
      }


    return score;
  }

  bool make_move(const int move_column) {
    for (int r = ROWS - 1; r >= 0; --r)
      if (board[r][move_column] == NONE) {
        board[r][move_column] = current_player;
        current_player = static_cast<Player>(-current_player);
        return true;
      }
    return false;
  }

  [[nodiscard]] __uint128_t hash() const {
    uint64_t yellow = 0;
    uint64_t red = 0;

    uint64_t flipped_yellow = 0;
    uint64_t flipped_red = 0;

    for (int r = 0; r < ROWS; ++r) {
      for (int c = 0; c < COLS; ++c) {
        if (board[r][c] == YELLOW) {
          yellow |= (1ULL << (r * COLS + c));
          flipped_yellow |= (1ULL << (r * COLS + (COLS - 1 - c)));
        } else if (board[r][c] == RED) {
          red |= (1ULL << (r * COLS + c));
          flipped_red |= (1ULL << (r * COLS + (COLS - 1 - c)));
        }
      }
    }

    if (flipped_yellow < yellow ||
        (flipped_yellow == yellow && flipped_red < red)) {
      yellow = flipped_yellow;
      red = flipped_red;
    }

    return (static_cast<__uint128_t>(yellow) << 64) | red;
  }

  [[nodiscard]] bool is_game_over() const { return legal_moves().empty(); }
};


std::unordered_map<__uint128_t, int32_t> transposition_table = {};


std::tuple<int32_t, int> minimax_impl(const GameState &game_state,
                                      const uint32_t depth = 9,
                                      int alpha = INT_MIN, int beta = INT_MAX) {
  // Check transposition table
  const __uint128_t game_hash = game_state.hash();
  const auto tt_it = transposition_table.find(game_hash);
  if (tt_it != transposition_table.end()) {
    return std::make_tuple(tt_it->second, -1);
  }

  // Terminal node check
  if (depth <= 0 || game_state.is_game_over()) {
    int32_t score = game_state.evaluate();
    return std::make_tuple(score, -1);
  }

  const bool is_max = game_state.current_player == RED;
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

MinimaxRet minimax(const int8_t board[ROWS][COLS], const int8_t player,
                   const int depth) {
  const GameState game_state(board, player);
  transposition_table.clear();
  transposition_table.reserve(1 << 18); // Reserve space
  auto [score, move] = minimax_impl(game_state, depth);
  return MinimaxRet{score, move};
}
}
