import gymnasium as gym
from famnit_gym.envs import mill
import random

### Let's play in a way to never capture any pieces. ###

env = mill.env(render_mode="human")
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    # The game should never terminate, but truncate after 100 moves.
    if truncation:
        print("The game was too long!")
        break

    # Here, we want to do some computations and we need the transition model.
    initial_model = mill.transition_model(env)

    # The transition model includes the following public methods:
    # * clone()
    #   Creates a deep copy of the object.
    #
    # * get_state()
    #   Returns the board position as a list [0 - 23], containing values:
    #   0 (empty), 1 (player 1), 2 (player 2).
    #   Note: Board position 1 has index 0 in the list, etc.
    #
    # * get_phase(player)
    #   Returns the phase of the given player (placing, moving, flying).
    #
    # * count_pieces(player)
    #   Returns number of pieces on the board belonging to the given player.
    #
    # * legal_moves(player)
    #   Returns the list of legal moves for the given player.
    #
    # * make_move(player, move)
    #   Changes the state as if the given player made the given move.
    #   Note: The correctnes of the move is not checked for performance
    #         reasons. The user should only make moves from the list of
    #         of legal moves. Player's turn is also not checked. The same
    #         player can be simulated as making multiple consecutive moves.
    #
    # * game_over()
    #   Return True if one of the player has lost the game.
    #
    # Printing the transition model prints the board state in ASCII.

    # We want to know which player's turn it is.
    player = 1 if agent == "player_1" else 2
    
    # And who is the opponent in this turn.
    opponent = 2 if player == 1 else 1

    # We will find all such moves that don't capture opponent's pieces.
    considered_moves = []

    # Try all possible moves of the current player.
    for move in initial_model.legal_moves(player=player):
        # Clone the existing model, so we can backtrack.
        model = initial_model.clone()
        
        # Check how many pieces the opponent has.
        pieces_count = model.count_pieces(player=opponent)
        
        # Make the move.
        model.make_move(player=player, move=move)
        
        # If no change in the opponent's count, we are fine with the move.
        if model.count_pieces(player=opponent) == pieces_count:
            considered_moves.append(move)
    
    # Choose the move randomly.
    move = considered_moves[random.randint(0, len(considered_moves) - 1)]

    # Make the move that we decided on.
    env.step(move)