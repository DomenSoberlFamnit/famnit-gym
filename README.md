# About

A collection of Gymnasium environments used with the Intelligent Systems course at UP FAMNIT.

# Installation

```console
pip install famnit_gym@git+https://github.com/DomenSoberlFamnit/famnit-gym
```

# Environments

## Sokoban

A single-agent Gymnasium environment that implements the Sokoban puzzle game.

A minimal example with random actions:

```python
import gymnasium as gym
import famnit_gym

# Create and reset the environment.
env = gym.make('famnit_gym/Sokoban-v1', render_mode='human')
observation, info = env.reset()

# Execute random actions.
done = False
while not done:
    action = env.action_space.sample()
    _, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
```

### Actions

| Integer | Meaning           |
|:-------:|-------------------|
| 0       | Move / push up    |
| 1       | Move / push right |
| 2       | Move / push down  |
| 3       | Move / push left  |

### Observations

A `height` × `width` numpy array, containing the following values:

| Integer | Meaning         |
|:-------:|-----------------|
| 0       | Floor           |
| 1       | Wall            |
| 2       | Crate           |
| 3       | Goal            |
| 4       | Crate on a goal |
| 5       | Player          |

### Available options

```python
options = {
    'map_template': map  # An integer 0 - 999 for a hardoded level, or a numpy array for a custom level.
    'scale': 0.75  # Scale the image when render_mode='human'.
}

env = gym.make('famnit_gym/Sokoban-v1', render_mode='human', options=options)
```

### Wrapper Keyboard

`famnit_gym.wrappers.sokoban.Keyboard`

Ignores the given action when calling the `step(action)` methods. Instead, it checks the keyboard input and executes the pressed key. If no key is pressed, the method call returns without an effect.

Available keys:
- UP: move / push up.
- RIGHT: move / push right.
- DOWN: move / push down.
- LEFT: move / push left.
- SPACE: Reset the episode.
- ESC: Truncate the episode.

### Wrapper Insights

`famnit_gym.wrappers.sokoban.Insights`

Expands the information given through the `info` variable:

```python
info = {
    'player': [x, y],                           # The location of the player.
    'crates': [[x0, y0], [x1, y1], ...],        # The locations of the crates.
    'goals': [[x0, y0], [x1, y1], ...],         # The locations of the goals.
    'finished': [[x0, y0], [x1, y1], ...],      # The locations of the crates on goal positions.
    'actions': {
        'moving': [0, ..., 3],                  # Actions currently effective for moving.
        'pushing': [0, ..., 3]                  # Actions currently effective for pushing.
    }
}
```
---

## Mill (Nine men's morris)

A multi-agent Petting Zoo environment that implements the game of Nine men's morris, also known as Mill. This implementation adds diagonal connections to the standard horizontal/vertical board map.

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
```

### Actions

An action is a list (or numpy array) `[src, dst, capture]`:

| Variable | Meaning                                                                                  |
|:---------|------------------------------------------------------------------------------------------|
| src      | The source position (1 - 24) of the piece or 0 when placing.                             |
| dst      | The destination position (1 - 24) of the piece.                                          |
| capture  | The position (1 - 24) from which the opponent's piece is captured or 0 if none captured. |

The list of legal actions is given in `info['legal_actions']`. Executing an illegal action or passing None to the `step()` method results in executing a random legal action.

### Observations

A numpy array with 24 integers (0 - 2), each representing the piece placed on the corresponding board position:

| Value | Meaning                                                                                  |
|:------|---------------------------------------|
| 0     | The position is empty.                |
| 1     | The position is occupied by player 1. |                                         |
| 2     | The position is occupied by player 2. |

### Info

```python
info = {
    'agent': 'player_1',             # player_1 or player_2
    'move': 1,                       # Increased after both player finished their turn.
    'phase': 'placing',              # Either placing, moving, flying, or lost.
    'legal_moves': [[0, 1, 0], ...]  # The list of currently legal moves.
}
```

### Wrapper Delay move

`famnit_gym.wrappers.mill.DelayMove`

Make the `step()` function block until the user presses a key, clicks a mouse, or a predetermined timer runs out.

### Wrapper User interaction

`famnit_gym.wrappers.mill.UserInteraction`

Allows interacting with the board by mouse or keyboard. Any board position can be marked with any color.

### Function transition_model

`famnit_gym.wrappers.mill.transition_model`

```python
model = transition_model(env)
```

Returns an object, independent of gymnasium, that the Mill environment uses internally to compute state transitions. It allows implementing off-line search argorithms without the need to implement the game logic separately.

# Examples

Examples of use can be found [here](https://github.com/DomenSoberlFamnit/famnit-gym/tree/main/famnit_gym/examples).

# License

`famnit-gym` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

# Acknowledgements

Sokoban tiles (CC licensed) are taken from [https://kenney.nl/assets/sokoban](https://kenney.nl/assets/sokoban).
