from typing import Optional
import numpy as np
import gym
from gym import spaces
import pyrosetta
from pyrosetta import *
from pyrosetta.teaching import *
from envs.bbo import BBO
import matplotlib.pyplot as plt
import io
from PIL import Image

pyrosetta.init()

### Generic continuous environment for reduced Hamiltonian dynamics framework
class Molecule(BBO):
    def __init__(self, pose, naive=False, reset_scale=90, step_size=1e-2, max_num_step=10, render_mode=None):
        # Superclass setup
        super(Molecule, self).__init__(naive, step_size, max_num_step)
        
        self.step_size = step_size 
        self.max_num_step = max_num_step
        self.num_step = 0
        self.render_mode = render_mode

        # Molecule info
        self.pose = pose
        self.num_residue = pose.total_residue()
        self.sfxn = get_fa_scorefxn() # Score function

        # State and action info
        self.state_dim = self.num_residue*2
        self.min_val = -180; self.max_val = 180
        #self.observation_space = spaces.Box(low=self.min_val, high=self.max_val, shape=(self.state_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=self.min_val, 
            high=self.max_val, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        self.min_act = -90; self.max_act = 90 
        self.action_space = spaces.Box(
            low=np.array([self.min_act] * self.state_dim), 
            high=np.array([self.max_act] * self.state_dim), 
            dtype=np.float32
        )
        #self.action_space = spaces.Box(low=self.min_act, high=self.max_act, shape=(self.state_dim,), dtype=np.float32)      
        self.state = None

        # Reset scale
        self.reset_scale = reset_scale

        # PyMol visualization
        self.pmm = PyMOLMover()
        self.pmm.keep_history(True)
    
    def step(self, action):
        self.state += self.step_size * action
        for k in range(self.num_residue):
            self.pose.set_phi(k+1, self.state[2*k]) 
            self.pose.set_psi(k+1, self.state[2*k+1])
        val = self.sfxn(self.pose)

        # Update number of step
        self.num_step += 1

        done = self.num_step >= self.max_num_step

        # Calculate final reward
        reward = self.calculate_final_reward(val, action)
        
        return np.array(self.state), reward, done, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.num_step = 0
        self.discount = 1.0
        return self.reset_at(mode='random')
    
    def reset_at(self, mode='random'):
        if mode == 'random':
            self.state = self.reset_scale*(self.rng.random(self.state_dim)-.5)
        elif mode == 'zero':
            # Set both phi and psi equal 0
            self.state = np.zeros(self.state_dim)
        return np.array(self.state)

    def render(self):
        for k in range(self.num_residue):
            self.pose.set_phi(k+1, self.state[2*k]) 
            self.pose.set_psi(k+1, self.state[2*k+1])
        self.pmm.apply(self.pose)

        if self.render_mode == "rgb_array":
            # Example placeholder: render a simple plot of phi/psi values
            phis = self.state[::2]
            psis = self.state[1::2]

            fig, ax = plt.subplots(figsize=(3, 3))
            ax.scatter(phis, psis, c='blue')
            ax.set_xlim(-180, 180)
            ax.set_ylim(-180, 180)
            ax.set_xlabel('Phi')
            ax.set_ylabel('Psi')
            ax.set_title('Ramachandran Plot')
            ax.grid(True)

            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            image = Image.open(buf).convert('RGB')
            return np.array(image)

        return None  # For render_mode=None or "human"
        
    def close(self):
        pass