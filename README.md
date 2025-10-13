# FAMNIT Gym

A collection of Gymnasium environments used with the Intelligent Systems course at UP FAMNIT.

## Installation

```console
pip install famnit_gym@git+https://github.com/DomenSoberlFamnit/famnit-gym
```

## Environments

### Sokoban

A minimum example with random actions:

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

#### Action codes

| Integer | Meaning           |
|:-------:|-------------------|
| 0       | Move / push up    |
| 1       | Move / push right |
| 2       | Move / push down  |
| 3       | Move / push left  |

#### Tile codes

| Integer | Meaning         |
|:-------:|-----------------|
| 0       | Floor           |
| 1       | Wall            |
| 2       | Crate           |
| 3       | Goal            |
| 4       | Crate on a goal |
| 5       | Player          |

#### Available options

```python
options = {
    'map_template': map  # An integer 0 - 999 for a hardoded level, or a numpy array for a custom level.
    'scale': 0.75  # Scale the image when render_mode='human'.
}

env = gym.make('famnit_gym/Sokoban-v1', render_mode='human', options=options)
```

#### Wrapper Keyboard

`famnit_gym.wrappers.sokoban.Keyboard`


Ignores the given action when calling the `step(action)` methods. Instead, it checks the keyboard input and executes the pressed key. If no key is pressed, the method call returns without an effect.

Available keys:
- UP: move / push up.
- RIGHT: move / push right.
- DOWN: move / push down.
- LEFT: move / push left.
- SPACE: Reset the episode.
- ESC: Truncate the episode.

#### Wrapper Insights

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

## License

`famnit-gym` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Acknowledgements

Sokoban tiles (CC licensed) are taken from [https://kenney.nl/assets/sokoban](https://kenney.nl/assets/sokoban).