import numpy as np
import imageio
import os

# Run num_traj trajectories using policy given by model on env.
def get_model_avg_final_vals(env, model, num_traj, num_step_per_traj,
                                 benchmark_model=False):
    vals = [] # List of last rewards for trajectories in test.
    for _ in range(num_traj):
        obs = env.reset() # obs same as state in our case.
        num_iteration = 0
        action = np.zeros(obs.shape)
        first_val = None
        for _ in range(num_step_per_traj):
            if benchmark_model:
                action, _ = model.predict(obs)
            else:
                action = model.get_action(obs, action)
            obs, reward, done, _ = env.step(action)
            if first_val is None:
                first_val = env.get_val(reward, action)
            num_iteration += 1
            if done:
                break
        val = env.get_val(reward, action)
        # print('First: {:.3f}, final: {:.3f}'.format(first_val, val))
        vals.append(val)

    return np.mean(np.array(vals))

# Visualize a particular trajectories from given model's policy.
def visualize(env, model, num_step=100, benchmark_model=False,
              extra_args='random', img_path=None):
    vals = []; images = []
    num_iteration = 0
    obs = env.reset_at(mode=extra_args)
    action = np.zeros(obs.shape)
    for _ in range(num_step):
        if benchmark_model:
            action, _ = model.predict(obs)
        else:
            action = model.get_action(obs, action)
        obs, reward, done, info = env.step(action)
        img = env.render()
        if img is not None:
            images.append(img)
        else:
            print(f"Skipping frame {num_iteration} – env.render() returned None")
        vals.append(env.get_val(reward, action))
        num_iteration += 1
        if done:
            break
    env.close()
    
    # Save simulation.
    if img_path is not None:
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        
        # Filter valid frames
        valid_frames = [np.array(img) for i, img in enumerate(images) if img is not None and i % 2 == 0]

        if len(valid_frames) == 0:
            print("⚠️ No valid frames to save. Skipping GIF creation.")
        else:
            imageio.mimsave(img_path, valid_frames, duration=50)
