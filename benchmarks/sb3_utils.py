import os, sys
import pandas as pd
import torch
from stable_baselines3 import PPO, SAC
from sb3_contrib import TRPO
from benchmarks.space_compatability_wrapper import SpaceCompatibilityWrapper
from utils import get_environment

# current directory
cur_dir = os.path.dirname(os.path.abspath(__file__))
# path to main repo (locally)
sys.path.append(os.path.dirname(cur_dir))
from utils import get_environment, str_to_list, DEVICE

def get_RL_nets_architectures(env_name, on_policy=True):
    # Get architecture info from arch_file
    arch_path = os.path.join(cur_dir, "arch.csv")
    df = pd.read_csv(arch_path)
    
    print("Columns in arch.csv:", df.columns)
    
    #strip whitespace from env_name column
    df['env_name'] = df['env_name'].str.strip()
    env_name = env_name.strip()
    
    #debug
    print(f"Looking for architecture of environment: '{env_name}'")
    print(f"Available envs in CSV: {df['env_name'].unique()}")
    
    net_info = df[df['env_name']==env_name]
    
    if net_info.empty:
        raise ValueError(f"Environment '{env_name}' not found in arch.csv.")
    
    print("Net info found:", net_info)
    
    actor_dims = str_to_list(net_info['actor_dims'].values[0])
    critic_dims_field = 'v_critic_dims' if on_policy else 'q_critic_dims'
    critic_dims = str_to_list(net_info[critic_dims_field].values[0])
    return actor_dims, critic_dims

def train_benchmark_model(method, gamma, env_name, total_samples, common_dims=[], 
          activation='ReLU', lr=3e-3, log_interval=10):
    # On-policy mean optimize policy directly using current policy being optimized.
    on_policy = method in set(['PPO', 'TRPO', 'A2C'])
    # off_policy = set('SAC', 'DDPG', 'TD3')

    # Construct environment
    env = get_environment(env_name)
    env.set_gamma(gamma) 
    actor_dims, critic_dims = get_RL_nets_architectures(env_name, on_policy=on_policy)

    # Net architecture for actor and critic networks
    if on_policy:
        net_arch_dict = dict(pi=actor_dims, vf=critic_dims)
    else:
        net_arch_dict = dict(pi=actor_dims, qf=critic_dims)
    
    # Add common processing nets from state to both actor & critic.
    if len(common_dims) != 0:
        net_arch = []
        for dim in common_dims:
            net_arch.append(dim)
        net_arch.append(net_arch_dict)
    else:
        net_arch = net_arch_dict
        
    # Set the policy args
    activation_fn = torch.nn.ReLU if activation == 'ReLU' else torch.nn.Tanh
    policy_kwargs = dict(activation_fn=activation_fn,
                         net_arch=net_arch)
    
    # Build the model using SB3.
    if method == 'TRPO':
        model = TRPO("MlpPolicy", env, learning_rate=lr, gamma=gamma, policy_kwargs=policy_kwargs, verbose=1)
    elif method == 'PPO':
        model = PPO("MlpPolicy", env, learning_rate=lr, gamma=gamma, policy_kwargs=policy_kwargs, verbose=1)
    elif method == 'SAC':
        model = SAC("MlpPolicy", env, learning_rate=lr, gamma=gamma, policy_kwargs=policy_kwargs, verbose=1)
    model.device = DEVICE

    # Train and save model with SB3.
    model.learn(total_timesteps=total_samples, log_interval=log_interval)
    model_path ="models/" + env_name + '_' + method + '_' + str(gamma).replace('.', '_')
    model.save(model_path)

def _load_benchmark_model(method, model_path, env):
    # Create a custom objects dictionary to override the observation_space
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space
    }
    
    if method.startswith('TRPO'):
        model = TRPO.load(model_path, env=env, custom_objects=custom_objects)
    elif method.startswith('PPO'):
        model = PPO.load(model_path, env=env, custom_objects=custom_objects)
    elif method.startswith('SAC'):
        model = SAC.load(model_path, env=env, custom_objects=custom_objects)
    return model

def setup_benchmark_model(method, env, env_name, model_version=''):
    wrapped_env = SpaceCompatibilityWrapper(env)  # Wrap the environment
    model_path = f"models/{env_name}_{method}{model_version}"
    
    model = _load_benchmark_model(method, model_path, wrapped_env)

    return model