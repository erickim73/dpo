from typing import Optional
import numpy as np
from scipy import interpolate
import cv2
from gym import spaces
from envs.bbo import BBO
import matplotlib.pyplot as plt
from PIL import Image
import io

from collections import namedtuple
ImgDim = namedtuple('ImgDim', 'width height')

class Shape(BBO):
    metadata = {'render.modes': ['human']}

    def __init__(self, naive=False, step_size=1e-2, state_dim=64, max_num_step=20, render_mode=None):
        # Superclass setup
        super(Shape, self).__init__(naive, step_size, max_num_step)

        # State and action info
        self.state_dim = state_dim
        self.max_val = 4; self.min_val = -4
        self.max_act = 1; self.min_act = -1
        self.action_space = spaces.Box(low=self.min_act, high=self.max_act, shape=(self.state_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.min_val, high=self.max_val, shape=(self.state_dim,), dtype=np.float32)
        self.state = None

        # Shape interpolation info
        self.xk, self.yk = np.mgrid[-1:1:8j, -1:1:8j]
        self.xg, self.yg = np.mgrid[-1:1:100j, -1:1:100j]
        self.viewer = ImgDim(width=100, height=100)
        
        self.step_size = step_size
        self.max_num_step = max_num_step
        
        self.render_mode = render_mode or "human"

    def step(self, action):
        self.state += self.step_size *action

        # Update number of step
        self.num_step += 1

        # Calculate value
        area, peri = geometry_info(self.state, self.xk, self.yk, self.xg, self.yg)
        done = (area == 0 or peri == 0)
        if not done:
            val = peri/np.sqrt(area)
            done = self.num_step >= self.max_num_step
        else:
            val = 1e9

        reward = self.calculate_final_reward(val, action)
        
        self.latest_reward = reward
        
        return np.array(self.state), reward, done, {}
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.num_step = 0
        self.discount = 1.0
        return self.reset_at(mode='random')
    
    def reset_at(self, mode='random'):
        self.num_step = 0
        width = int(np.sqrt(self.state_dim))
        self.state = np.ones((width, width))
        if mode=='hole':
            self.state[1:8, 1:8] = 0
            self.state += self.rng.random((width, width))
            self.state = np.clip(self.state, 0, 1)
        elif mode=='random':
            self.state = self.rng.random((width, width))
        elif mode=='random_with_padding':
            # Random with zero padding
            self.state[1:(width-1), :(width-1)] = self.rng.random((width-2, width-1))
        self.state -= .5
        self.state = self.state.reshape(-1)
        return np.array(self.state)
    
    def render(self):
        try:
            xk, yk, xg, yg = self.xk, self.yk, self.xg, self.yg
            z = self.state.reshape(xk.shape[0], yk.shape[0])

            # Generate binary image
            binary = spline_interp(z, xk, yk, xg, yg)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create white canvas (larger for better quality)
            canvas_size = 512
            canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

            if contours:
                # Get bounding box for scaling and centering
                all_pts = np.vstack(contours).squeeze()
                x, y, w, h = cv2.boundingRect(all_pts)
                scale = 0.9 * canvas_size / max(w, h)
                offset_x = (canvas_size - scale * w) / 2
                offset_y = (canvas_size - scale * h) / 2

                # Transform and draw contours
                transformed = [np.int32(scale * (cnt - [x, y]) + [offset_x, offset_y]) for cnt in contours]
                cv2.drawContours(canvas, transformed, -1, (0, 0, 0), 2)

            return canvas

        except Exception as e:
            print(f"[Shape.render] Failed to render frame: {e}")
            return None


## Helper functions ##
# Spline interpolation for 2D density problem
def spline_interp(z, xk, yk, xg, yg):
    # Interpolate knots with bicubic spline
    tck = interpolate.bisplrep(xk, yk, z)
    
    # Evaluate bicubic spline on (fixed) grid
    zint = interpolate.bisplev(xg[:,0], yg[0,:], tck)
    
    # zint is between [-1, 1]
    zint = np.clip(zint, -1, 1)
    
    # ✅ Rescale to [0, 255] and apply binary threshold
    img = ((zint + 1) * 127.5).astype('uint8')  # [-1,1] to [0,255]
    _, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return thresh_img
    

def geometry_info_from_img(img):
    # Extract contours and calculate perimeter/area   
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0; peri = 0
    for cnt in contours:
        area -= cv2.contourArea(cnt, oriented=True)
        peri += cv2.arcLength(cnt, closed=True)
    
    return area, peri

def geometry_info(z, xk, yk, xg, yg):
    img = spline_interp(z, xk, yk, xg, yg)
    return geometry_info_from_img(img)