from gym.envs.registration import register

register(
    id='swarm-v0',
    entry_point='gym_swarm.envs:SwarmEnv',
    max_episode_steps=1000,
)