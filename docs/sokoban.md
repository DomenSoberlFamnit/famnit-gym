# Sokoban

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

# Close the environment.
env.close()
```

## Actions

| Integer | Meaning           |
|:-------:|-------------------|
| 0       | Move / push up    |
| 1       | Move / push right |
| 2       | Move / push down  |
| 3       | Move / push left  |

## Observations

A `height` × `width` numpy array of `uint8` values, indexed as `observation[y][x]`, containing the following values:

| Integer | Meaning         |
|:-------:|-----------------|
| 0       | Floor           |
| 1       | Wall            |
| 2       | Crate           |
| 3       | Goal            |
| 4       | Crate on a goal |
| 5       | Player          |

The player (5) hides the tile underneath it, so a player standing on a goal is shown as 5.

Wall tiles that are completely surrounded by other walls are turned into floor (0) when the level is loaded. In the built-in levels, most of the area outside the outer wall is therefore shown as floor, although the player can never reach it.

## Info

```python
info = {
    'steps': 0  # The number of steps made in the episode.
}
```

## Rewards

The reward is 1 when all crates are on goals, otherwise 0.

## Episode end

The episode is terminated when all crates are on goals. It is truncated when the user closes the window. There is no step limit.

## Available options

```python
options = {
    'map_template': map,  # An integer 0 - 999 for a hardcoded level, or a numpy array for a custom level.
    'scale': 0.75  # Scale the image when render_mode='human'.
}

env = gym.make('famnit_gym/Sokoban-v1', render_mode='human', options=options)
```

## Wrapper Keyboard

`famnit_gym.wrappers.sokoban.Keyboard`

Ignores the given action when calling the `step(action)` method. Instead, it checks the keyboard input and executes the pressed key. If no key is pressed, the method call returns without an effect. Switches the environment to `render_mode='human'`.

Available keys:
- UP: move / push up.
- RIGHT: move / push right.
- DOWN: move / push down.
- LEFT: move / push left.
- SPACE: Reset the episode.
- ESC: Truncate the episode.

Closing the window also truncates the episode.

## Wrapper Insights

`famnit_gym.wrappers.sokoban.Insights`

Expands the information given through the `info` variable:

```python
info = {
    'player': [x, y],                           # The location of the player.
    'crates': [[x0, y0], [x1, y1], ...],        # The locations of the crates.
    'goals': [[x0, y0], [x1, y1], ...],         # The locations of the goals without crates.
    'finished': [[x0, y0], [x1, y1], ...],      # The locations of the crates on goal positions.
    'actions': {
        'moving': [0, ..., 3],                  # Actions currently effective for moving.
        'pushing': [0, ..., 3]                  # Actions currently effective for pushing.
    }
}
```

## Wrapper Video

`famnit_gym.wrappers.sokoban.Video`

Records the episode and saves the video to a file when `close()` is called. Switches the environment to `render_mode='human'`.

```python
from famnit_gym.wrappers.sokoban import Video

env = Video(env, filename='sokoban.webm')

env.close()  # Saves the episode recording to the file.
```

The `filename` parameter is optional and defaults to `'sokoban.mp4'`. Both `.mp4` and `.webm` files are supported.
