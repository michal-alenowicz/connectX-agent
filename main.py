import os, sys, random
import numpy as np
import pickle

cwd = '/kaggle_simulations/agent/'
if os.path.exists(cwd):
    sys.path.append(cwd)
else:
    cwd = ''

opening_book = None

def my_agent7(obs, config):
    

    
    
    def load_opening_book(filename="/kaggle_simulations/agent/opening_book.pkl"):
        """Load dictionary from disk."""
        import pickle
        with open(filename, "rb") as f:
            return pickle.load(f)
    
    # Gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows-1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid

    # Helper function for get_heuristic: checks if window satisfies heuristic conditions
    def check_window(window, num_discs, piece, config):
        return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)
    
    # Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
    def count_windows(grid, num_discs, piece, config):
        num_windows = 0
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        return num_windows

    
    #function to identify the "hot" locations, i.e. exact fields (one per col) that may be filled in the next move
    def get_grid_hot(grid, config):
        valid_moves = [c for c in range(config.columns) if grid[0][c] == 0]
        grid_hot = grid.copy()
        for col in valid_moves:
            for row in range(config.rows-1, -1, -1):
                if grid_hot[row][col] == 0:
                    break
            grid_hot[row][col] = 9
        #print(grid_hot)
        return grid_hot

    #checks for given number of discs combined with HOT fields
    def check_window_for_hot(window, num_discs, piece, config):
        return (window.count(piece) == num_discs and window.count(9) == config.inarow-num_discs)

    #checks for particularly BADASS situation of two same-color discs sandwiched between two HOT fields
    def check_window_for_badass_two(window, piece, config):
        return (window == [9,piece,piece,9])

    #counts HOT windows
    def count_hot_windows(grid, num_discs, piece, config):
        num_windows = 0
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                if check_window_for_hot(window, num_discs, piece, config):
                    num_windows += 1
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                if check_window_for_hot(window, num_discs, piece, config):
                    num_windows += 1
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if check_window_for_hot(window, num_discs, piece, config):
                    num_windows += 1
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if check_window_for_hot(window, num_discs, piece, config):
                    num_windows += 1 
        return num_windows

    
    #counts BADASS TWO windows
    def count_badass_two_windows(grid, piece, config):
        num_windows = 0
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                if check_window_for_badass_two(window, piece, config):
                    num_windows += 1
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                if check_window_for_badass_two(window, piece, config):
                    num_windows += 1
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if check_window_for_badass_two(window, piece, config):
                    num_windows += 1
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if check_window_for_badass_two(window, piece, config):
                    num_windows += 1
        return num_windows

    #checks two-field window for two connected pieces (meant for the center column)
    def check_center_window_for_any_two(window, piece):
        return (window.count(piece) == 2)
    
    #counts any vertical connected two in the center col (3) using 2-field windows
    def count_any_center_two(grid, piece, config):
        num_windows = 0
        # vertical
        for row in range(config.rows-1):
            for col in [3]:
                window = list(grid[row:row+2, col])
                if check_center_window_for_any_two(window, piece):
                    num_windows += 1
        return num_windows

    #evaluates the position by multiplying the grid by positional matrix (for one given player mark!)
    def positional_matrix_eval(grid, mark, config):
        board_points = [3, 4, 5, 7, 5, 4, 3, 4, 6, 8, 10, 8, 6, 4,5, 8, 11, 13, 11, 8, 5,5, 8, 11, 13, 11, 8, 5, 4, 6, 8, 10, 8, 6, 4, 3, 4, 5, 7, 5, 4, 3]
        grid_points = np.asarray(board_points).reshape(config.rows, config.columns)
        evaluation = 0
        for row in range(config.rows):
            for col in range(config.columns):
                if grid[row][col] == mark:
                    evaluation += grid_points[row][col]
        return evaluation
    
        # HEURISTIC MOŻNA MODYFIKOWAĆ Helper function for score_move: calculates value of heuristic for grid
    def get_heuristic (grid, mark, config):

        grid_hot = get_grid_hot(grid, config)
        
        #num_twos = count_windows(grid, 2, mark, config)
        num_threes = count_windows(grid, 3, mark, config)
        num_fours = count_windows(grid, 4, mark, config)
        num_hot_threes = count_hot_windows(grid_hot, 3, mark, config)
        num_badass_twos = count_badass_two_windows(grid_hot, mark, config)
        positional_eval = positional_matrix_eval(grid, mark, config)
        #num_twos_opp = count_windows(grid, 2, mark%2+1, config)
        num_threes_opp = count_windows(grid, 3, mark%2+1, config)
        num_fours_opp = count_windows(grid, 4, mark%2+1, config)
        num_hot_threes_opp = count_hot_windows(grid_hot, 3, mark%2+1, config)
        num_badass_twos_opp = count_badass_two_windows(grid_hot, mark%2+1, config)
        positional_eval_opp = positional_matrix_eval(grid, mark%2+1, config)
        #tylko dla naszego gracza:
        num_center_twos = count_any_center_two(grid, mark, config)
        
        score = 200000000*num_fours + 15000*num_hot_threes + 4000*num_badass_twos + 500*num_threes + 1*num_center_twos + 1*positional_eval + (-1*positional_eval_opp) + (-1000*num_threes_opp) + (-7000*num_badass_twos_opp) + (-50000*num_hot_threes_opp) + (-200000000*num_fours_opp)
        return score


    # Uses ALPHABETA to calculate value of dropping piece in selected column
    def score_move(grid, col, mark, config, nsteps):
        next_grid = drop_piece(grid, col, mark, config)
        score = alphabeta(next_grid, nsteps-1, -np.inf, +np.inf, False, mark, config)
        return score

    # Helper function for minimax/alphabeta: checks if agent or opponent has four in a row in the window
    def is_terminal_window(window, config):
        return window.count(1) == config.inarow or window.count(2) == config.inarow

    # Helper function for minimax/alphabeta: checks if game has ended
    def is_terminal_node(grid, config):
        # Check for draw 
        if list(grid[0, :]).count(0) == 0:
            return True
        # Check for win: horizontal, vertical, or diagonal
        # horizontal 
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                if is_terminal_window(window, config):
                    return True
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                if is_terminal_window(window, config):
                    return True
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if is_terminal_window(window, config):
                    return True
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if is_terminal_window(window, config):
                    return True
        return False

    # ALPHABETA implementation
    def alphabeta(node, depth, alpha, beta, maximizingPlayer, mark, config):
        is_terminal = is_terminal_node(node, config)
        valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
        if depth == 0 or is_terminal:
            return get_heuristic(node, mark, config)
        if maximizingPlayer:
            value = -np.inf
            for col in valid_moves:
                child = drop_piece(node, col, mark, config)
                value = max(value, alphabeta(child, depth-1,alpha, beta, False, mark, config))
                if value >= beta:
                    break
                alpha = max(alpha, value)
            return value
        else:
            value = np.inf
            for col in valid_moves:
                child = drop_piece(node, col, mark%2+1, config)
                value = min(value, alphabeta(child, depth-1, alpha, beta, True, mark, config))
                if value <= alpha:
                    break
                beta = min(beta, value)
            return value

    #MAIN
    #start_time = time.time()

    global opening_book
    # load the book
    if opening_book == None:
        opening_book = load_opening_book()
    
    N_STEPS = 3
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    #score_current = get_heuristic(grid,obs.mark,config)
    #print('Ocena bieżącej pozycji=',score_current)

    # Flatten and make a tuple for dictionary
    board_key = (tuple(grid.flatten()), obs.mark)

    if board_key in opening_book:
        return opening_book[board_key]
    else:
        # Use the heuristic (WITH N_STEPS) to assign a score to each possible board in the next turn
        scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))
        # Get a list of columns (moves) that maximize the heuristic
        #print(get_grid_hot(grid, config))
        #print(scores)
        max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
        # Select THE CENTER-MOST from the maximizing columns!
        #print ('agent4 think time elapsed=',(time.time()-start_time))
        return min(max_cols, key=lambda c: abs(c - (config.columns // 2)))
