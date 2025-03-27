import numpy as np
import random
import time
import pygame
import sys

# Initialize pygame
pygame.init()

# Colors
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)

# Game dimensions
SQUARESIZE = 100
COLUMN_COUNT = 7
ROW_COUNT = 6
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE
size = (width, height)
RADIUS = int(SQUARESIZE/2 - 5)

# Font
FONT = pygame.font.SysFont("monospace", 75)

class Connect4:
    def __init__(self, width=7, height=6):
        self.width = width
        self.height = height
        self.board = np.zeros((height, width), dtype=int)
        self.player = 1  # Player 1 starts (1 for player, 2 for AI)
        
    def drop_piece(self, col, piece):
        """Drop a piece into the specified column."""
        for row in range(self.height-1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = piece
                return row
        return -1  # Column is full
    
    def is_valid_location(self, col):
        """Check if a column is valid for dropping a piece."""
        return self.board[0][col] == 0
    
    def get_valid_locations(self):
        """Get all valid columns for dropping a piece."""
        return [col for col in range(self.width) if self.is_valid_location(col)]
    
    def check_win(self, piece):
        """Check if the specified player has won."""
        # Check horizontal
        for r in range(self.height):
            for c in range(self.width-3):
                if all(self.board[r][c+i] == piece for i in range(4)):
                    return True
        
        # Check vertical
        for r in range(self.height-3):
            for c in range(self.width):
                if all(self.board[r+i][c] == piece for i in range(4)):
                    return True
        
        # Check diagonal (positive slope)
        for r in range(self.height-3):
            for c in range(self.width-3):
                if all(self.board[r+i][c+i] == piece for i in range(4)):
                    return True
        
        # Check diagonal (negative slope)
        for r in range(3, self.height):
            for c in range(self.width-3):
                if all(self.board[r-i][c+i] == piece for i in range(4)):
                    return True
        
        return False
    
    def is_terminal_node(self):
        """Check if the game is over."""
        return self.check_win(1) or self.check_win(2) or len(self.get_valid_locations()) == 0

def draw_board(game, screen, first_player):
    # Draw the board background
    for c in range(game.width):
        for r in range(game.height):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
    
    # Draw the pieces
    for c in range(game.width):
        for r in range(game.height):
            if game.board[r][c] == 1:  # Player 1 (X or O depending on who starts)
                if first_player == 0:  # User starts, so user is X
                    draw_x(screen, c, r)
                else:  # AI starts, so user is O
                    draw_o(screen, c, r)
            elif game.board[r][c] == 2:  # Player 2 (X or O depending on who starts)
                if first_player == 0:  # User starts, so AI is O
                    draw_o(screen, c, r)
                else:  # AI starts, so AI is X
                    draw_x(screen, c, r)
    
    pygame.display.update()

def draw_x(screen, col, row):
    # Draw X
    x_pos = int(col*SQUARESIZE+SQUARESIZE/2)
    y_pos = int(row*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)
    offset = RADIUS * 0.7
    
    pygame.draw.line(screen, RED, (x_pos-offset, y_pos-offset), (x_pos+offset, y_pos+offset), 10)
    pygame.draw.line(screen, RED, (x_pos+offset, y_pos-offset), (x_pos-offset, y_pos+offset), 10)

def draw_o(screen, col, row):
    # Draw O
    x_pos = int(col*SQUARESIZE+SQUARESIZE/2)
    y_pos = int(row*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)
    
    pygame.draw.circle(screen, YELLOW, (x_pos, y_pos), RADIUS-5, 10)

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

def minimax(game, depth, maximizing_player):
    """
    Minimax algorithm implementation for Connect-4.
    """
    valid_locations = game.get_valid_locations()
    is_terminal = game.is_terminal_node()
    
    if depth == 0 or is_terminal:
        if is_terminal:
            if game.check_win(2):  # AI wins
                return (None, 1000000)
            elif game.check_win(1):  # Player wins
                return (None, -1000000)
            else:  # Game is a draw
                return (None, 0)
        else:  # Depth is zero
            return (None, score_position(game.board, 2))
    
    if maximizing_player:
        value = -float('inf')
        column = random.choice(valid_locations)
        for col in valid_locations:
            game_copy = Connect4(game.width, game.height)
            game_copy.board = game.board.copy()
            game_copy.drop_piece(col, 2)
            new_score = minimax(game_copy, depth-1, False)[1]
            if new_score > value:
                value = new_score
                column = col
        return column, value
    
    else:  # Minimizing player
        value = float('inf')
        column = random.choice(valid_locations)
        for col in valid_locations:
            game_copy = Connect4(game.width, game.height)
            game_copy.board = game.board.copy()
            game_copy.drop_piece(col, 1)
            new_score = minimax(game_copy, depth-1, True)[1]
            if new_score < value:
                value = new_score
                column = col
        return column, value

def play_game_minimax_visual():
    """Play a visual game of Connect-4 against the AI using Minimax."""
    # Setup pygame display
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption('Connect 4 - Minimax')
    
    # Create start screen
    screen.fill(BLACK)
    title_font = pygame.font.SysFont("monospace", 50)
    option_font = pygame.font.SysFont("monospace", 30)
    
    title_text = title_font.render("Connect 4 with Minimax", True, WHITE)
    start_text = option_font.render("Who starts first?", True, WHITE)
    user_text = option_font.render("0: You (X)", True, WHITE)
    ai_text = option_font.render("1: AI (X)", True, WHITE)
    
    screen.blit(title_text, (width//2 - title_text.get_width()//2, height//4))
    screen.blit(start_text, (width//2 - start_text.get_width()//2, height//2))
    screen.blit(user_text, (width//2 - user_text.get_width()//2, height//2 + 50))
    screen.blit(ai_text, (width//2 - ai_text.get_width()//2, height//2 + 100))
    
    # Draw rectangles around clickable options
    pygame.draw.rect(screen, GRAY, (width//2 - user_text.get_width()//2 - 10, height//2 + 45, 
                                   user_text.get_width() + 20, user_text.get_height() + 10), 2)
    pygame.draw.rect(screen, GRAY, (width//2 - ai_text.get_width()//2 - 10, height//2 + 95, 
                                   ai_text.get_width() + 20, ai_text.get_height() + 10), 2)
    
    pygame.display.update()
    
    # Wait for user input to determine who starts
    first_player = None
    while first_player is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0 or event.key == pygame.K_KP0:
                    first_player = 0  # User starts
                elif event.key == pygame.K_1 or event.key == pygame.K_KP1:
                    first_player = 1  # AI starts
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Get click position
                posx, posy = event.pos
                # Check if user clicked on "You (X)" option
                if (width//2 - user_text.get_width()//2 - 10 <= posx <= width//2 + user_text.get_width()//2 + 10 and
                    height//2 + 45 <= posy <= height//2 + 45 + user_text.get_height() + 10):
                    first_player = 0  # User starts
                # Check if user clicked on "AI (X)" option
                elif (width//2 - ai_text.get_width()//2 - 10 <= posx <= width//2 + ai_text.get_width()//2 + 10 and
                      height//2 + 95 <= posy <= height//2 + 95 + ai_text.get_height() + 10):
                    first_player = 1  # AI starts
    
    # Setup game
    game = Connect4()
    game_over = False
    turn = first_player  # 0 for user, 1 for AI
    
    # Draw initial board
    screen.fill(BLACK)
    draw_board(game, screen, first_player)
    
    # Start game timer
    start_time = time.time()
    
    # Main game loop
    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEMOTION and turn == 0:  # User's turn
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                posx = event.pos[0]
                
                # Show piece above board
                if first_player == 0:  # User is X
                    draw_x(screen, posx//SQUARESIZE, -0.5)
                else:  # User is O
                    draw_o(screen, posx//SQUARESIZE, -0.5)
                    
                pygame.display.update()
            
            if event.type == pygame.MOUSEBUTTONDOWN and turn == 0:  # User's turn
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                
                # Get user's move
                posx = event.pos[0]
                col = posx // SQUARESIZE
                
                if col >= 0 and col < game.width and game.is_valid_location(col):
                    game.drop_piece(col, 1)
                    
                    if game.check_win(1):
                        label = FONT.render("You win!", 1, RED if first_player == 0 else YELLOW)
                        screen.blit(label, (40, 10))
                        game_over = True
                    
                    # Switch turns
                    turn = 1
                    
                    # Update board
                    draw_board(game, screen, first_player)
        
        # AI's turn
        if turn == 1 and not game_over:
            # Show thinking message
            pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
            thinking_font = pygame.font.SysFont("monospace", 30)
            thinking_text = thinking_font.render("AI is thinking...", True, WHITE)
            screen.blit(thinking_text, (width//2 - thinking_text.get_width()//2, SQUARESIZE//2 - thinking_text.get_height()//2))
            pygame.display.update()
            
            # Get AI's move
            col, minimax_score = minimax(game, 4, True)  # Depth of 4
            
            if game.is_valid_location(col):
                # Add a small delay to make AI's move visible
                pygame.time.wait(500)
                game.drop_piece(col, 2)
                
                if game.check_win(2):
                    label = FONT.render("AI wins!", 1, YELLOW if first_player == 0 else RED)
                    screen.blit(label, (40, 10))
                    game_over = True
                
                # Switch turns
                turn = 0
                
                # Update board
                draw_board(game, screen, first_player)
        
        # Check for draw
        if len(game.get_valid_locations()) == 0 and not game_over:
            label = FONT.render("Draw!", 1, WHITE)
            screen.blit(label, (40, 10))
            game_over = True
            draw_board(game, screen, first_player)
    
    # Calculate total game time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Display total time
    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
    time_font = pygame.font.SysFont("monospace", 30)
    
    if game.check_win(1):
        result_text = "You win!"
    elif game.check_win(2):
        result_text = "AI wins!"
    else:
        result_text = "Draw!"
        
    result_label = time_font.render(result_text, True, WHITE)
    time_label = time_font.render(f"Total time: {total_time:.2f} seconds", True, WHITE)
    
    screen.blit(result_label, (width//2 - result_label.get_width()//2, 10))
    screen.blit(time_label, (width//2 - time_label.get_width()//2, 50))
    pygame.display.update()
    
    # Wait for user to close the window
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

if __name__ == "__main__":
    play_game_minimax_visual()
