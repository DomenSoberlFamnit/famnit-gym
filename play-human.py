import sys
import random
from famnit_gym.envs import mill
from famnit_gym.wrappers.mill.user_interaction import UserInteraction
from mcts import mcts_search

DIFFICULTIES = {
    "Easy":   {"iterations": 25,  "depth": 10, "blunder_rate": 0.20},
    "Medium": {"iterations": 400, "depth": 40, "blunder_rate": 0.0},
    "Hard":   {"iterations": 1000, "depth": 100, "blunder_rate": 0.0},
}

def choose_ai_move(model, player, difficulty_name="Easy"):
    """Chooses a move for the AI according to the selected difficulty."""
    cfg = DIFFICULTIES[difficulty_name]
    legal_moves = model.legal_moves(player=player)

    if not legal_moves:
        return None

    # Check for blunder for AI on easy mode
    if random.random() < cfg["blunder_rate"]:
        return random.choice(legal_moves)

    return mcts_search(
        root_model=model,
        player=player,
        iterations=cfg["iterations"],
        max_simulation_depth=cfg["depth"]
    )

def get_human_move(env, model, player):
    legal_moves = model.legal_moves(player=player)
    phase = model.get_phase(player)
    
    # Track board positions clicked
    clicks = []
    
    # Color for highlighting selected piece (Light Green)
    SELECT_COLOR = (50, 220, 50, 200)

    print(f"\n[Your Turn - Phase: {phase.upper()}] Click on the board!")
    if phase == "placing":
        print("  -> Click an empty circle to place your piece.")
    else:
        print("  -> 1st Click: select piece to move. 2nd Click: destination.")

    while True:
        event = env.interact()

        if event.get("type") == "quit":
            env.close()
            sys.exit(0)

        if event.get("type") == "mouse_click":
            pos = event.get("position")
            if pos == 0:
                continue  

            clicks.append(pos)
            env.mark_position(pos, SELECT_COLOR)


            if phase == "placing":
                candidate = [0, clicks[0], 0]
                if len(clicks) == 1 and candidate in legal_moves:
                    env.clear_markings()
                    return candidate
                
                if len(clicks) == 2:
                    candidate = [0, clicks[0], clicks[1]]
                    if candidate in legal_moves:
                        env.clear_markings()
                        return candidate
                    else:
                        print(f"Cannot remove piece at position {clicks[1]}. Try another opponent piece.")
                        env.unmark_position(clicks[1])
                        clicks.pop()
                elif len(clicks) == 1:
                    mill_moves = [m for m in legal_moves if m[0] == 0 and m[1] == clicks[0]]
                    if mill_moves:
                        print(f"🔥 Mill formed! Click an opponent's piece to remove it.")
                    else:
                        print(f"Position {pos} is not a legal placement. Try again.")
                        env.clear_markings()
                        clicks = []

            else:
                if len(clicks) == 2:
                    candidate = [clicks[0], clicks[1], 0]
                    if candidate in legal_moves:
                        env.clear_markings()
                        return candidate
                    mill_moves = [m for m in legal_moves if m[0] == clicks[0] and m[1] == clicks[1]]
                    if mill_moves:
                        print(f"🔥 Mill formed! Click an opponent's piece to remove it.")
                    else:
                        print(f"Illegal move from {clicks[0]} to {clicks[1]}. Try again.")
                        env.clear_markings()
                        clicks = []

                elif len(clicks) == 3:
                    candidate = [clicks[0], clicks[1], clicks[2]]
                    if candidate in legal_moves:
                        env.clear_markings()
                        return candidate
                    else:
                        print(f"Cannot remove piece at position {clicks[2]}. Try another piece.")
                        env.unmark_position(clicks[2])
                        clicks.pop()

raw_env = mill.env(render_mode="human")
env = UserInteraction(raw_env)
env.reset()

# Settings to be changed by the user:
HUMAN_PLAYER = "player_1"   # Player 1 or Player 2, Player 1 goes first
AI_DIFFICULTY = "Hard"      # "Easy", "Medium", or "Hard" depending on how challenging you want the AI to be

print("=" * 60)
print(f"Game has begun: Human ({HUMAN_PLAYER}) vs AI ({AI_DIFFICULTY})")
print("=" * 60)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        print("\n Game Over!")
        if env.rewards["player_1"] > 0:
            print("Victory! You won!")
        elif env.rewards["player_2"] > 0:
            print("Defeat! The AI has beat you. Shame.")
        else:
            print("Draw! It is what it is. Nobody wins.")
        break

    model = mill.transition_model(raw_env)
    player = 1 if agent == "player_1" else 2

    if agent == HUMAN_PLAYER:
        move = get_human_move(env, model, player)
    else:
        print(f"\nAI ({AI_DIFFICULTY}) is thinking...")
        move = choose_ai_move(model, player, difficulty_name=AI_DIFFICULTY)

    env.step(move)

env.close()