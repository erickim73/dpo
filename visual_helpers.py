#helps visualize shape during training process
import os
import imageio
import cv2

def save_render(env, step_idx, output_dir, reward=None, loss=None):
    frame = env.render()
    if frame is not None:
        os.makedirs(output_dir, exist_ok=True)

        # Convert to BGR for OpenCV drawing
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Add overlay text
        overlay_text = f"Step: {step_idx}"
        if reward is not None:
            overlay_text += f" | Reward: {reward:.3f}"
        if loss is not None:
            overlay_text += f" | Loss: {loss:.3f}"

        cv2.putText(frame_bgr, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Convert back to RGB for saving
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imageio.imwrite(f"{output_dir}/frame_{step_idx:04d}.png", frame_rgb)


def make_video_from_frames(frame_dir, output_path="evolution.mp4", fps=10):
    """
    Compiles a folder of PNG frames into a single video.
    """
    filenames = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.endswith(".png")
    ])
    frames = [imageio.imread(f) for f in filenames]
    imageio.mimsave(output_path, frames, fps=fps)
