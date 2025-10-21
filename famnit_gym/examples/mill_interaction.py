import gymnasium as gym
from famnit_gym.envs import mill
from famnit_gym.wrappers.mill import UserInteraction

env = mill.env(render_mode='human')

# Wrap the Mill environment into the UserInteraction wrapper
# to enable the user to interact with the board.
env = UserInteraction(env)
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination:
        print(f"{agent} lost the game!")
        break

    if truncation:
        print("The game is too long!")
        break

    # Here, the user interacts with the board for a while.
    # In this example, the interaction is done when the user
    # clicks on an empty position on the board.
    done_interacting = False
    while not done_interacting:
        # Blocks, until one of the following events occurs:
        # - The user quits (closes the window).
        # - A mouse is move off a position (0) or onto a new position (1 - 24).
        # - A mouse button has been clicked.
        # - A key has been pressed.
        event = env.interact()

        # If the user quit, let us truncate the game.
        if event["type"] == "quit":
            done_interacting = True
            truncation = True
        
        # Use a different selection color for empty and occupied positions.
        elif event["type"] == "mouse_move":
            if observation[event["position"] - 1] == 0:
                env.set_selection_color(64, 192, 0, 128)  # RGBA for empty.
            else:
                env.set_selection_color(128, 128, 255, 255)  # RGBA for occupied.
        
        # Only if clicked on an empty position, the game continues.
        elif event["type"] == "mouse_click":
            if observation[event["position"] - 1] == 0:
                done_interacting = True

        # If escape key has been pressed, the game is truncated.
        elif event["type"] == "key_press":
            if event["key"] == "escape":
                done_interacting = True
                truncation = True
    
    # Id truncated during the interaction, the user quit interactively.
    if truncation:
        print("User quit interactively!")
        break

    # When done with interaction, a random move is executed.
    env.step(None)