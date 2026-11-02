# Mill (Nine men's morris)

A multi-agent PettingZoo environment that implements the game of Nine men's morris, also known as Mill. This implementation adds diagonal connections to the standard horizontal/vertical board map.

A minimal example with random actions:

```python
import gymnasium as gym
from famnit_gym.envs import mill

# Create and reset the environment.
env = mill.env(render_mode='human')  # 'ansi' for ASCII board.
env.reset()

# Execute random actions.
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination:
        print(f"{agent} lost the game!")
        break

    if truncation:
        print("The game is too long!")
        break
    
    env.step(None)

# Close the environment.
env.close()
```

## Board positions

The board positions are numbered 1 - 24:

```
1---------2---------3
| \       |       / |
|   4-----5-----6   |
|   | \   |   / |   |
|   |   7-8-9   |   |
10--11--12  13--14--15
|   |  16-17-18 |   |
|   | /   |   \ |   |
|   19----20----21  |
| /       |       \ |
22--------23--------24
```

## Actions

An action is a list (or numpy array) `[src, dst, capture]`:

| Variable | Meaning                                                                                  |
|:---------|------------------------------------------------------------------------------------------|
| src      | The source position (1 - 24) of the piece or 0 when placing.                             |
| dst      | The destination position (1 - 24) of the piece.                                          |
| capture  | The position (1 - 24) from which the opponent's piece is captured or 0 if none captured. |

The list of legal actions is given in `info['legal_moves']`. Executing an illegal action or passing None to the `step()` method results in executing a random legal action.

## Observations

A numpy array of 24 integers (`uint8`, 0 - 2), each representing the piece placed on the corresponding board position:

| Value | Meaning                               |
|:------|---------------------------------------|
| 0     | The position is empty.                |
| 1     | The position is occupied by player 1. |
| 2     | The position is occupied by player 2. |

## Info

```python
info = {
    'agent': 'player_1',             # player_1 or player_2
    'move': 1,                       # Increased after both players finish their turn.
    'phase': 'placing',              # Either placing, moving, flying, or lost.
    'legal_moves': [[0, 1, 0], ...]  # A numpy array of currently legal moves.
}
```

## Rewards

When a player captures a piece, it receives a reward of 1 and the opponent receives -1. Otherwise both receive 0.

## Episode end

The game is terminated when a player loses, which happens when they are left with only two pieces or cannot make a legal move. The game is truncated after 100 moves, or when the user closes the window.

## Wrapper DelayMove

`famnit_gym.wrappers.mill.DelayMove`

Makes the `step()` method wait until the user presses a key, clicks the mouse, or `time_limit` seconds pass (default 5). Requires `render_mode='human'`. Closing the window truncates the episode.

```python
from famnit_gym.wrappers.mill import DelayMove

env = DelayMove(env, time_limit=10)
```

## Wrapper UserInteraction

`famnit_gym.wrappers.mill.UserInteraction`

Allows interacting with the board by mouse or keyboard. Any board position can be marked with any color. Requires `render_mode='human'`. Its methods:

- `interact()` blocks until the user moves the mouse to another position, clicks, presses a key, or closes the window, and returns a message: `{'type': 'mouse_move', 'position': p}`, `{'type': 'mouse_click', 'position': p}`, `{'type': 'key_press', 'key': name}`, or `{'type': 'quit'}`. The position is 0 when the mouse is not over any board position.
- `mark_position(position, color)`, `unmark_position(position)` and `clear_markings()` highlight board positions. The color is an RGBA tuple.
- `set_selection_color(color)` sets the color used to highlight the position under the mouse.

```python
from famnit_gym.wrappers.mill import UserInteraction

env = UserInteraction(env)

env.mark_position(5, (128, 128, 0, 128))  # Mark position 5 with a yellow shade.
message = env.interact()                  # Wait for the user.
```

## Wrapper Video

`famnit_gym.wrappers.mill.Video`

Saves the video to a file after the episode has finished. Requires `render_mode='human'`.

```python
from famnit_gym.wrappers.mill import Video

env = Video(env, filename='mill.webm')

env.close()  # Saves the episode recording to the file.
```

The `filename` parameter is optional and defaults to `'mill.webm'`.

## Function transition_model

`famnit_gym.envs.mill.transition_model`

```python
from famnit_gym.envs import mill

model = mill.transition_model(env.unwrapped)
```

Returns an object, independent of Gymnasium, that the Mill environment uses internally to compute state transitions. It allows implementing off-line search algorithms without the need to implement the game logic separately.

The model has these methods. `player` is 1 or 2:

| Method                    | Meaning                                                      |
|---------------------------|--------------------------------------------------------------|
| `legal_moves(player)`     | Returns the list of legal moves of the player.               |
| `make_move(player, move)` | Executes the move `[src, dst, capture]` on the model and returns a dictionary with the resulting phases and piece counts. |
| `clone()`                 | Returns an independent copy of the model.                    |
| `count_pieces(player)`    | Returns the number of the player's pieces on the board.      |
| `get_phase(player)`       | Returns the player's phase: placing, moving, flying, or lost. |
| `get_state()`             | Returns the board as a list of 24 values, as in observations. |
| `game_over()`             | Returns True if one of the players has lost.                 |
