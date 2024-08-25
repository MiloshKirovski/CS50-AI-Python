"""
Tic Tac Toe Player
"""
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)

    return O if x_count > o_count else X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))

    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action
    copied_board = copy.deepcopy(board)

    if not (0 <= i <= 2 and 0 <= j <= 2):
        raise Exception(f"Action out of bounds: ({i}, {j})!")
    if board[i][j] is not EMPTY:
        raise Exception(f"Cell ({i}, {j}) is already taken!")

    player_turn = player(board)
    copied_board[i][j] = player_turn

    return copied_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] is not EMPTY:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] is not EMPTY:
            return board[0][i]

    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not EMPTY:
        return board[0][0]
    if board[2][0] == board[1][1] == board[0][2] and board[2][0] is not EMPTY:
        return board[2][0]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True

    for row in board:
        if EMPTY in row:
            return False

    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    winner_of_board = winner(board)
    return 1 if winner_of_board == X else -1 if winner_of_board == O else 0


def min_value(board):
    if terminal(board):
        return utility(board)
    v = float('inf')
    for action in actions(board):
        v = min(v, max_value(result(board, action)))

    return v


def max_value(board):
    if terminal(board):
        return utility(board)
    v = float('-inf')
    for action in actions(board):
        v = max(v, min_value(result(board, action)))

    return v


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    current_player = player(board)

    def optimal_value(is_maximizing):
        if terminal(board):
            return None

        best_value = float('-inf') if is_maximizing else float('inf')
        best_action = None

        for action in actions(board):
            result_board = result(board, action)
            if winner(result_board) == current_player:
                return action

            value = max_value(result_board) if is_maximizing else min_value(result_board)

            if (is_maximizing and value > best_value) or (not is_maximizing and value < best_value):
                best_value = value
                best_action = action

        return best_action

    return optimal_value(current_player == X)
