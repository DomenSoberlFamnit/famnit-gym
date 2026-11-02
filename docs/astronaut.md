# Astronaut

A single-agent Gymnasium environment in which an astronaut with a jetpack flies through a side-scrolling cave. The astronaut must avoid hitting the ground and the ceiling by switching the jetpack thrust on and off. Optional wrappers add asteroids to avoid and oxygen tanks to collect.

The simulation runs at a fixed 60 frames per second, so each call to `step()` advances the simulation by 1/60 of a second. The game area is 1200 × 800 pixels, with the `y` coordinate measured upwards from the bottom of the screen.

A minimal example with random actions:

```python
import gymnasium as gym
import famnit_gym

# Create and reset the environment.
env = gym.make('famnit_gym/Astronaut-v1', render_mode='human')
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

Use `render_mode=None` for a headless simulation at maximum speed.

## Actions

| Integer | Meaning        |
|:-------:|----------------|
| 0       | Thrust off     |
| 1       | Thrust on      |

## Observations

A numpy array of `float32` values. The base environment observes only the astronaut:

| Index | Variable      | Meaning                                          |
|:-----:|---------------|--------------------------------------------------|
| 0     | `astronaut_y` | The vertical position of the astronaut (pixels). |
| 1     | `astronaut_v` | The vertical velocity of the astronaut (pixels / second). |

The `Asteroids` and `Oxygen` wrappers append their own variables to the end of the array, in the order the wrappers are applied. The names of all variables, in the order they appear in the observation, are given in `info['variables']`:

```python
i = info['variables'].index('asteroid_y')
asteroid_y = observation[i]
```

## Episode end

The episode is **terminated** when the astronaut collides with the ground, the ceiling, or an asteroid (with the `Asteroids` wrapper). The episode is **truncated** after `max_steps` steps.

## Info

```python
info = {
    'variables': ['astronaut_y', 'astronaut_v'],  # The names of the observation variables.
    'steps': 0,                                   # The number of steps made in the episode.
    'collision': 'no'                             # Either no, ground, ceiling, or asteroid.
}
```

## Rewards

The default reward function returns:

| Reward | Event                                                |
|:------:|------------------------------------------------------|
| -1     | Collision.                                           |
| 1      | Oxygen tank picked (only with the `Oxygen` wrapper). |
| 0      | Otherwise.                                           |

A custom reward function can be given through the `reward_function` option. It receives a dictionary with the following entries and returns the reward for the step:

```python
reward_info = {
    'collision': False,     # True if the astronaut has collided.
    'oxygen_picked': False  # True if an oxygen tank was picked in this step (only with the Oxygen wrapper).
}
```

## Available options

```python
options = {
    'astronaut_x': 300,      # The horizontal position of the astronaut (pixels).
    'speed': 500,            # The scrolling speed (pixels / second).
    'thrust': 1000,          # The upward acceleration of the jetpack (pixels / second squared).
    'gravity': 400,          # The downward acceleration of gravity (pixels / second squared).
    'max_steps': 1200,       # The number of steps after which the episode is truncated.
    'reward_function': None  # A custom reward function, or None for the default one.
}

env = gym.make('famnit_gym/Astronaut-v1', render_mode='human', options=options)
```

All options are optional; the values above are the defaults. Gravity always applies, so while thrusting the net upward acceleration is `thrust - gravity`. For the astronaut to be able to rise, `thrust` must be greater than `gravity`.

Options can also be passed to `reset()`. They are combined with the default values, not with the options given to `gym.make`.

## Wrapper Asteroids

`famnit_gym.wrappers.astronaut.Asteroids`

Adds asteroids flying towards the astronaut at random heights. A collision with an asteroid terminates the episode. Appends the position of the next approaching asteroid to the observation:

| Variable     | Meaning                                         |
|--------------|-------------------------------------------------|
| `asteroid_x` | The horizontal position of the asteroid (pixels). |
| `asteroid_y` | The vertical position of the asteroid (pixels).   |

## Wrapper Oxygen

`famnit_gym.wrappers.astronaut.Oxygen`

Adds oxygen tanks for the astronaut to collect at random heights. Appends the position of the next approaching oxygen tank to the observation:

| Variable   | Meaning                                              |
|------------|------------------------------------------------------|
| `oxygen_x` | The horizontal position of the oxygen tank (pixels). |
| `oxygen_y` | The vertical position of the oxygen tank (pixels).   |

Expands the information given through the `info` variable:

```python
info = {
    'oxygen_picked': 'no',  # 'yes' if an oxygen tank was picked in this step.
    'oxygen_count': 0       # The number of oxygen tanks picked in the episode.
}
```

## Wrapper DebugOverlay

`famnit_gym.wrappers.astronaut.DebugOverlay`

Draws additional information on the screen when `render_mode='human'`: the collision boundaries, the positions of the astronaut, asteroids and oxygen tanks, the cumulative reward, the number of steps, and the number of collected oxygen tanks.

## Wrapper Keyboard

`famnit_gym.wrappers.astronaut.Keyboard`

Ignores the given action when calling the `step(action)` method. Instead, it checks the keyboard input and executes the pressed key. Requires `render_mode='human'` and should be applied as the last (top) wrapper.

Available keys:
- SPACE: Thrust (hold).
- ENTER: Reset the episode.
- P: Pause / resume the game.
- ESC: Truncate the episode.

## Wrapper Video

`famnit_gym.wrappers.astronaut.Video`

Saves the video to a file after the episode has finished.

```python
from famnit_gym.wrappers.astronaut import Video

env = Video(env, filename='astronaut.webm')

env.close()  # Saves the episode recording to the file.
```

The `filename` parameter is optional and defaults to `'astronaut.webm'`. Both `.webm` and `.mp4` files are supported.

## Stacking wrappers

The wrappers can be combined:

```python
import gymnasium as gym
import famnit_gym
from famnit_gym.wrappers.astronaut import Asteroids, Oxygen, DebugOverlay, Video, Keyboard

env = gym.make('famnit_gym/Astronaut-v1', render_mode='human')
env = Asteroids(env)
env = Oxygen(env)
env = DebugOverlay(env)
env = Video(env, filename='astronaut.webm')
env = Keyboard(env)  # Keyboard must be the top wrapper.
```
