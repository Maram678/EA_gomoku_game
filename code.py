import numpy as np
import sys
import random
import tkinter as tk
from tkinter import messagebox

# Define constants
EMPTY = 0
BLACK = 1
WHITE = 2
ROW_COUNT = 15
COLUMN_COUNT = 15
SQUARESIZE = 30
RADIUS = SQUARESIZE // 2 - 2
WINDOW_LENGTH = 5
WINDOW_WIDTH = COLUMN_COUNT * SQUARESIZE
WINDOW_HEIGHT = (ROW_COUNT + 1) * SQUARESIZE

# Neural Network parameters
input_size = ROW_COUNT * COLUMN_COUNT
hidden_size = 100
output_size = COLUMN_COUNT

# Neural Network class
class NeuralNetwork:
    def _init_(self):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden)
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.output = self.sigmoid(self.output_layer_input)
        return self.output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Genetic Algorithm class
class GeneticAlgorithm:
    def _init_(self, population_size):
        self.population_size = population_size
        self.black_population = [NeuralNetwork() for _ in range(population_size)]
        self.white_population = [NeuralNetwork() for _ in range(population_size)]
        self.black_wins = np.zeros(population_size)
        self.white_wins = np.zeros(population_size)

    def calculate_distance(self, individual1, individual2):
        return np.linalg.norm(individual1.weights_input_hidden - individual2.weights_input_hidden) \
               + np.linalg.norm(individual1.weights_hidden_output - individual2.weights_hidden_output)

    def calculate_adjusted_fitness(self, wins, population):
        adjusted_fitness = np.zeros(len(population))
        for i, individual in enumerate(population):
            distance_sum = sum(self.calculate_distance(individual, other) for other in population)
            adjusted_fitness[i] = wins[i] / distance_sum
        return adjusted_fitness

    def evolve(self):
        black_adjusted_fitness = self.calculate_adjusted_fitness(self.black_wins, self.black_population)
        white_adjusted_fitness = self.calculate_adjusted_fitness(self.white_wins, self.white_population)

        new_black_population = []
        new_white_population = []

        for _ in range(self.population_size):
            parent1 = random.choices(self.black_population, weights=black_adjusted_fitness)[0]
            parent2 = random.choices(self.black_population, weights=black_adjusted_fitness)[0]
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_black_population.append(child)

        for _ in range(self.population_size):
            parent1 = random.choices(self.white_population, weights=white_adjusted_fitness)[0]
            parent2 = random.choices(self.white_population, weights=white_adjusted_fitness)[0]
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_white_population.append(child)

        self.black_population = new_black_population
        self.white_population = new_white_population

    def crossover(self, parent1, parent2):#REAL_VALUED |FLOATING-POINT REPRESENTATION #INTERMDIATE CROSSOVER #SINGLE
        child = NeuralNetwork()

        # Single arithmetic crossover
        crossover_point = np.random.randint(1, input_size * hidden_size + hidden_size * output_size)#This expression calculates the total number of weights in the neural network.
        flattened_parent1 = np.concatenate((parent1.weights_input_hidden.flatten(), parent1.weights_hidden_output.flatten()))
        flattened_parent2 = np.concatenate((parent2.weights_input_hidden.flatten(), parent2.weights_hidden_output.flatten()))
        child_weights = np.zeros_like(flattened_parent1)

        child_weights[:crossover_point] = flattened_parent1[:crossover_point]
        child_weights[crossover_point:] = flattened_parent2[crossover_point:]

        child.weights_input_hidden = child_weights[:input_size * hidden_size].reshape(input_size, hidden_size)
        child.weights_hidden_output = child_weights[input_size * hidden_size:].reshape(hidden_size, output_size)

        return child

    def mutate(self, individual):
        mutation_rate = 0.01

        for layer in [individual.weights_input_hidden, individual.weights_hidden_output]:
            mutation_mask = np.random.rand(*layer.shape) < mutation_rate
            mutation_values = np.random.uniform(low=-0.1, high=0.1, size=layer.shape)
            layer[mutation_mask] += mutation_values[mutation_mask]

        return individual

# Create the game board
def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)

# Check if a move is valid
def is_valid_location(board, col):
    return 0 <= col < COLUMN_COUNT and board[ROW_COUNT - 1][col] == 0

# Get the next open row in a column
def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r

# Drop a piece into the board
def drop_piece(board, row, col, piece):
    board[row][col] = piece

# Check for a win
def winning_move(board, piece):
    for c in range(COLUMN_COUNT - WINDOW_LENGTH + 1):#0=11
        for r in range(ROW_COUNT):#15
            if np.all(board[r, c:c + WINDOW_LENGTH] == piece):
                return True

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - WINDOW_LENGTH + 1):
            if np.all(board[r:r + WINDOW_LENGTH, c] == piece):
                return True

    for c in range(COLUMN_COUNT - WINDOW_LENGTH + 1):
        for r in range(ROW_COUNT - WINDOW_LENGTH + 1):
            if np.all(board[r:r + WINDOW_LENGTH, c:c + WINDOW_LENGTH].diagonal() == piece):
                return True

    for c in range(COLUMN_COUNT - WINDOW_LENGTH + 1):
        for r in range(ROW_COUNT - 1, WINDOW_LENGTH - 2, -1):
            if np.all(np.fliplr(board)[r - WINDOW_LENGTH + 1:r + 1, c:c + WINDOW_LENGTH].diagonal() == piece):
                return True

    return False

def draw_board(board):
    canvas.delete("all")  # Clear the canvas
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            canvas.create_rectangle(c * SQUARESIZE, r * SQUARESIZE, (c + 1) * SQUARESIZE, (r + 1) * SQUARESIZE, outline="black")
            if board[r][c] == BLACK:
                canvas.create_oval(c * SQUARESIZE, r * SQUARESIZE, (c + 1) * SQUARESIZE, (r + 1) * SQUARESIZE, fill="black")
            elif board[r][c] == WHITE:
                canvas.create_oval(c * SQUARESIZE, r * SQUARESIZE, (c + 1) * SQUARESIZE, (r + 1) * SQUARESIZE, fill="white")
    root.update()


# Main game loop
def main():
    # Store seeds
    seeds = []

    # Loop for 30 runs
    for _ in range(5):
        seed = random.randint(1, 1000)  # Generate a random seed
        seeds.append(seed)  # Store the seed

        # Initialize random number generator with the seed
        random.seed(seed)
        np.random.seed(seed)

        # Create the game board
        board = create_board()
        game_over = False
        turn = BLACK

        # Initialize Genetic Algorithm
        population_size = 20
        genetic_algorithm = GeneticAlgorithm()

        while not game_over:
            if turn == BLACK:
                # Black player's turn
                current_board_state = board.flatten()
                current_board_state[current_board_state == BLACK] = 1
                current_board_state[current_board_state == WHITE] = -1
                current_board_state[current_board_state == EMPTY] = 0
                current_board_state = current_board_state.reshape(1, -1)

                # Get AI move using neural network from black population
                neural_network = random.choice(genetic_algorithm.black_population)
                output = neural_network.forward(current_board_state)
                col = np.argmax(output)

                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, BLACK)
                    if winning_move(board, BLACK):
                        game_over = True
                        genetic_algorithm.black_wins += 1
                    turn = WHITE

            else:
                # White player's turn
                current_board_state = board.flatten()# row=0 ,cols=n=255
                current_board_state[current_board_state == BLACK] = -1
                current_board_state[current_board_state == WHITE] = 1
                current_board_state[current_board_state == EMPTY] = 0
                current_board_state = current_board_state.reshape(1, -1)

                # Get AI move using neural network from white population
                neural_network = random.choice(genetic_algorithm.white_population)
                output = neural_network.forward(current_board_state)
                col = np.argmax(output)

                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, WHITE)
                    if winning_move(board, WHITE):
                        game_over = True
                        genetic_algorithm.white_wins += 1
                    turn = BLACK

            # Draw the board
            draw_board(board)

        # Display winning message after winning move
        if winning_move(board, BLACK):
            print("Run {}: Black wins!".format(_ + 1))
            
        elif winning_move(board, WHITE):
            print("Run {}: White wins!".format(_ + 1))
            


    # Print the list of seeds
    print("Seeds used for initialization:", seeds)

# Initialize Tkinter
root = tk.Tk()
root.title("Gomoku")
canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
canvas.pack()

# Start the game
main()

root.mainloop()