import numpy as np
import random
import time
import matplotlib.pyplot as plt
import sys
from connect_4_minimax_1 import Connect4, minimax
from connect_4_ab_pruning_1 import alpha_beta_pruning

def minimax_with_counter(game, depth, maximizing_player, node_count=0):
    """Wrapper for minimax that counts nodes."""
    node_count += 1
    valid_locations = game.get_valid_locations()
    is_terminal = game.is_terminal_node()
    
    if depth == 0 or is_terminal:
        if is_terminal:
            if game.check_win(2):  # AI wins
                return (None, 1000000, node_count)
            elif game.check_win(1):  # Player wins
                return (None, -1000000, node_count)
            else:  # Game is a draw
                return (None, 0, node_count)
        else:  # Depth is zero
            return (None, score_position(game.board, 2), node_count)
    
    if maximizing_player:
        value = -float('inf')
        column = random.choice(valid_locations)
        for col in valid_locations:
            game_copy = Connect4(game.width, game.height)
            game_copy.board = game.board.copy()
            game_copy.drop_piece(col, 2)
            _, new_score, new_count = minimax_with_counter(game_copy, depth-1, False, node_count)
            node_count = new_count
            if new_score > value:
                value = new_score
                column = col
        return column, value, node_count
    
    else:  # Minimizing player
        value = float('inf')
        column = random.choice(valid_locations)
        for col in valid_locations:
            game_copy = Connect4(game.width, game.height)
            game_copy.board = game.board.copy()
            game_copy.drop_piece(col, 1)
            _, new_score, new_count = minimax_with_counter(game_copy, depth-1, True, node_count)
            node_count = new_count
            if new_score < value:
                value = new_score
                column = col
        return column, value, node_count

def alpha_beta_with_counter(game, depth, alpha, beta, maximizing_player, node_count=0):
    """Wrapper for alpha-beta pruning that counts nodes."""
    node_count += 1
    valid_locations = game.get_valid_locations()
    is_terminal = game.is_terminal_node()
    
    if depth == 0 or is_terminal:
        if is_terminal:
            if game.check_win(2):  # AI wins
                return (None, 1000000, node_count)
            elif game.check_win(1):  # Player wins
                return (None, -1000000, node_count)
            else:  # Game is a draw
                return (None, 0, node_count)
        else:  # Depth is zero
            return (None, score_position(game.board, 2), node_count)
    
    if maximizing_player:
        value = -float('inf')
        column = random.choice(valid_locations)
        for col in valid_locations:
            game_copy = Connect4(game.width, game.height)
            game_copy.board = game.board.copy()
            game_copy.drop_piece(col, 2)
            _, new_score, new_count = alpha_beta_with_counter(game_copy, depth-1, alpha, beta, False, node_count)
            node_count = new_count
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Beta cutoff
        return column, value, node_count
    
    else:  # Minimizing player
        value = float('inf')
        column = random.choice(valid_locations)
        for col in valid_locations:
            game_copy = Connect4(game.width, game.height)
            game_copy.board = game.board.copy()
            game_copy.drop_piece(col, 1)
            _, new_score, new_count = alpha_beta_with_counter(game_copy, depth-1, alpha, beta, True, node_count)
            node_count = new_count
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break  # Alpha cutoff
        return column, value, node_count

def score_position(board, piece):
    """Score the entire board position for the given piece."""
    score = 0
    height, width = board.shape
    
    # Score center column (preferred strategy)
    center_array = [int(board[r][width//2]) for r in range(height)]
    center_count = center_array.count(piece)
    score += center_count * 3
    
    # Score horizontal
    for r in range(height):
        row_array = [int(board[r][c]) for c in range(width)]
        for c in range(width-3):
            window = row_array[c:c+4]
            score += evaluate_window(window, piece)
    
    # Score vertical
    for c in range(width):
        col_array = [int(board[r][c]) for r in range(height)]
        for r in range(height-3):
            window = col_array[r:r+4]
            score += evaluate_window(window, piece)
    
    # Score positive diagonal
    for r in range(height-3):
        for c in range(width-3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    
    # Score negative diagonal
    for r in range(height-3):
        for c in range(width-3):
            window = [board[r+3-i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
    
    return score

def evaluate_window(window, piece):
    """Evaluate a window of 4 positions for the given piece."""
    opponent_piece = 1 if piece == 2 else 2
    
    if window.count(piece) == 4:
        return 100
    elif window.count(piece) == 3 and window.count(0) == 1:
        return 5
    elif window.count(piece) == 2 and window.count(0) == 2:
        return 2
    elif window.count(opponent_piece) == 3 and window.count(0) == 1:
        return -4
    
    return 0

def generate_random_board(move_count=10):
    """Generate a random board with some moves played."""
    game = Connect4()
    for _ in range(move_count):
        valid_moves = game.get_valid_locations()
        if not valid_moves:
            break
        col = random.choice(valid_moves)
        piece = random.choice([1, 2])
        game.drop_piece(col, piece)
    return game

def benchmark_depth_comparison():
    """Compare performance at different depths."""
    depths = [1, 2, 3, 4, 5]
    num_positions = 5
    
    minimax_times = {depth: [] for depth in depths}
    alpha_beta_times = {depth: [] for depth in depths}
    minimax_nodes = {depth: [] for depth in depths}
    alpha_beta_nodes = {depth: [] for depth in depths}
    
    print("Running depth comparison benchmark...")
    
    for i in range(num_positions):
        print(f"Testing position {i+1}/{num_positions}")
        game = generate_random_board(random.randint(5, 15))
        
        for depth in depths:
            # Test Minimax
            start_time = time.time()
            _, _, nodes = minimax_with_counter(game, depth, True)
            end_time = time.time()
            minimax_times[depth].append(end_time - start_time)
            minimax_nodes[depth].append(nodes)
            
            # Test Alpha-Beta
            start_time = time.time()
            _, _, nodes = alpha_beta_with_counter(game, depth, -float('inf'), float('inf'), True)
            end_time = time.time()
            alpha_beta_times[depth].append(end_time - start_time)
            alpha_beta_nodes[depth].append(nodes)
    
    # Print results
    print("\n--- DEPTH COMPARISON RESULTS ---")
    print("Depth | Minimax Time | Alpha-Beta Time | Speedup | Minimax Nodes | Alpha-Beta Nodes | Node Reduction")
    print("-" * 100)
    
    for depth in depths:
        avg_minimax_time = sum(minimax_times[depth]) / len(minimax_times[depth])
        avg_alpha_beta_time = sum(alpha_beta_times[depth]) / len(alpha_beta_times[depth])
        avg_minimax_nodes = sum(minimax_nodes[depth]) / len(minimax_nodes[depth])
        avg_alpha_beta_nodes = sum(alpha_beta_nodes[depth]) / len(alpha_beta_nodes[depth])
        
        speedup = avg_minimax_time / avg_alpha_beta_time if avg_alpha_beta_time > 0 else float('inf')
        node_reduction = 100 * (1 - avg_alpha_beta_nodes / avg_minimax_nodes) if avg_minimax_nodes > 0 else 0
        
        print(f"{depth:5d} | {avg_minimax_time:12.6f}s | {avg_alpha_beta_time:15.6f}s | {speedup:7.2f}x | {avg_minimax_nodes:13.0f} | {avg_alpha_beta_nodes:16.0f} | {node_reduction:13.1f}%")
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Time comparison
    plt.subplot(2, 1, 1)
    plt.title('Execution Time Comparison')
    plt.xlabel('Search Depth')
    plt.ylabel('Average Time (seconds)')
    
    avg_minimax_times = [sum(minimax_times[depth]) / len(minimax_times[depth]) for depth in depths]
    avg_alpha_beta_times = [sum(alpha_beta_times[depth]) / len(alpha_beta_times[depth]) for depth in depths]
    
    plt.plot(depths, avg_minimax_times, 'o-', label='Minimax')
    plt.plot(depths, avg_alpha_beta_times, 's-', label='Alpha-Beta')
    plt.legend()
    plt.grid(True)
    
    # Node comparison
    plt.subplot(2, 1, 2)
    plt.title('Nodes Explored Comparison')
    plt.xlabel('Search Depth')
    plt.ylabel('Average Nodes Explored')
    
    avg_minimax_nodes_list = [sum(minimax_nodes[depth]) / len(minimax_nodes[depth]) for depth in depths]
    avg_alpha_beta_nodes_list = [sum(alpha_beta_nodes[depth]) / len(alpha_beta_nodes[depth]) for depth in depths]
    
    plt.plot(depths, avg_minimax_nodes_list, 'o-', label='Minimax')
    plt.plot(depths, avg_alpha_beta_nodes_list, 's-', label='Alpha-Beta')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png')
    print("Comparison plot saved as 'algorithm_comparison.png'")

def max_depth_in_time_limit():
    """Find maximum depth each algorithm can reach in a time limit."""
    time_limits = [0.1, 0.5, 1.0, 2.0, 5.0]
    num_positions = 3
    
    print("\nTesting maximum depth within time limits...")
    print("Time Limit | Minimax Max Depth | Alpha-Beta Max Depth | Depth Advantage")
    print("-" * 75)
    
    for time_limit in time_limits:
        minimax_max_depths = []
        alpha_beta_max_depths = []
        
        for i in range(num_positions):
            game = generate_random_board(random.randint(5, 10))
            
            # Test Minimax
            max_depth = 0
            for depth in range(1, 10):
                start_time = time.time()
                try:
                    minimax_with_counter(game, depth, True)
                    end_time = time.time()
                    if end_time - start_time > time_limit:
                        break
                    max_depth = depth
                except:
                    break
            minimax_max_depths.append(max_depth)
            
            # Test Alpha-Beta
            max_depth = 0
            for depth in range(1, 15):  # Alpha-Beta can go deeper
                start_time = time.time()
                try:
                    alpha_beta_with_counter(game, depth, -float('inf'), float('inf'), True)
                    end_time = time.time()
                    if end_time - start_time > time_limit:
                        break
                    max_depth = depth
                except:
                    break
            alpha_beta_max_depths.append(max_depth)
        
        avg_minimax_max_depth = sum(minimax_max_depths) / len(minimax_max_depths)
        avg_alpha_beta_max_depth = sum(alpha_beta_max_depths) / len(alpha_beta_max_depths)
        depth_advantage = avg_alpha_beta_max_depth - avg_minimax_max_depth
        
        print(f"{time_limit:10.1f}s | {avg_minimax_max_depth:18.1f} | {avg_alpha_beta_max_depth:21.1f} | {depth_advantage:15.1f}")

if __name__ == "__main__":
    print("Connect-4 Algorithm Benchmark")
    print("============================")
    print("Comparing Minimax vs Alpha-Beta Pruning")
    print("Current date:", time.strftime("%A, %B %d, %Y, %I:%M %p"))
    print("============================\n")
    
    benchmark_start_time = time.time()
    
    benchmark_depth_comparison()
    max_depth_in_time_limit()
    
    benchmark_end_time = time.time()
    total_benchmark_time = benchmark_end_time - benchmark_start_time
    
    print(f"\nTotal benchmark time: {total_benchmark_time:.2f} seconds")
    print("Benchmark complete!")
