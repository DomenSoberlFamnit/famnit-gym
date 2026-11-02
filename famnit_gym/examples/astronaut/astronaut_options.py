import gymnasium as gym
import famnit_gym

# A custom reward function.
reward_function = lambda reward_info: (
    -1 if reward_info.get('collision', False)  # Collision.
    else 1 if reward_info.get('oxygen_picked', False)  # Oxygen picked (available only with the Oxygen wrapper).
    else 0.1  #  Successful flight.
)

# Custom environment parameters. FPS = 60 is fixed.
options = {
    'astronaut_x': 600,  # pixels (default 300)
    'speed': 800,        # pixels / second (default 500)
    'thrust': 350,       # pixels / second squared (default 1000)
    'gravity': 200,      # pixels / second squared (default 400)
    'max_steps': 300,    # When to truncate the episode (default 1200)
    'reward_function': reward_function  # A custom reward function (default None)
}

# Use render_mode = None for headless simulation and maximum speed.
render_mode = None
#render_mode = 'human'

# Create and reset the environment.
env = gym.make('famnit_gym/Astronaut-v1', render_mode=render_mode, options=options)
observation, info = env.reset()

# Try to fly for 5 seconds and earn 30 reward points.
episode_reward = 0
while True:
    action = env.action_space.sample()  # Random action.

    # Make actions 6 times longer (100 milliseconds).
    action_reward = 0
    for _ in range(6):
        observation, reward, terminated, truncated, info = env.step(action)
        action_reward += reward

    episode_reward += action_reward

    if terminated:
        print("Crashed.")
        break

    if truncated:
        print("Success.")
        break

print(f"Episode reward: {episode_reward:.1f}")

# Close the environment.
env.close()