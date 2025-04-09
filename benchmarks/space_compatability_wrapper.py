from gym import Wrapper
import numpy as np
from gym import spaces
from sb3_contrib import TRPO
from stable_baselines3 import PPO, SAC

class SpaceCompatibilityWrapper(Wrapper):
    """Wrapper to ensure observation and action spaces match expected format."""
    def __init__(self, env):
        super().__init__(env)
        
        # Match observation space format
        obs_dim = env.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=np.array([-180.0] * obs_dim),
            high=np.array([180.0] * obs_dim),
            dtype=np.float32
        )
        
        # Match action space format
        act_dim = env.action_space.shape[0]
        self.action_space = spaces.Box(
            low=np.array([-90.0] * act_dim),
            high=np.array([90.0] * act_dim),
            dtype=np.float32
        )

def setup_benchmark_model(method, env, env_name, model_version=''):
    # Wrap the environment to ensure space compatibility
    wrapped_env = SpaceCompatibilityWrapper(env)
    
    model_path = "models/" + env_name + '_' + method + model_version
    
    # Use wrapped_env for loading the model
    if method == 'TRPO':
        model = TRPO.load(model_path, env=wrapped_env)
    elif method == 'PPO':
        model = PPO.load(model_path, env=wrapped_env)
    elif method == 'SAC':
        model = SAC.load(model_path, env=wrapped_env)
    
    # We don't set_env here, we keep using the wrapped env
    
    return model