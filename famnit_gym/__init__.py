from gymnasium.envs.registration import register

register(
    id="famnit_gym/Sokoban-v1",
    entry_point="famnit_gym.envs:SokobanEnv",
)

register(
    id="famnit_gym/Astronaut-v1",
    entry_point="famnit_gym.envs:AstronautEnv",
)