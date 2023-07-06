# coding: utf-8
"""Defines some frozen lake maps."""
import gymnasium as gym

from gymnasium.envs.registration import register

# De-register environments if there is a collision
env_dict = gym.envs.registration.registry.copy()
for env in env_dict:
    if "Deterministic-4x4-FrozenLake-v0" in env:
        del gym.envs.registration.registry[env]
    elif "Stochastic-4x4-FrozenLake-v0" in env:
        del gym.envs.registration.registry[env]
    elif "TaxiEnv-v3" in env:
        del  gym.envs.registration.registry[env]
    elif "CartPoleEnv-v0" in env:
        del  gym.envs.registration.registry[env]
    elif 'Pendulum-v1' in env:
        del  gym.envs.registration.registry[env]


register(
    id="Deterministic-4x4-FrozenLake-v0",
    entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": False},
)

register(
    id="Stochastic-4x4-FrozenLake-v0",
    entry_point="gymnasium.envs.toy_text.frozen_lake:FrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": True},
)

register(
    id="TaxiEnv-v3",
    entry_point="gymnasium.envs.toy_text.taxi:TaxiEnv"    
)


register(
    id="CartPoleEnv-v0",
    entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv"    
)

register(
    id="Pendulum-v1",
    entry_point="gymnasium.envs.classic_control.pendulum:PendulumEnv"    
)

