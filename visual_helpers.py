#helps visualize shape during training process
import os
import imageio
import cv2
import numpy as np

def save_render(env, step_idx, output_dir, reward=None, loss=None):
    frame = env.render()
    if frame is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        if len(frame.shape) == 2:  # (H, W)
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[2] == 1:  # (H, W, 1)
            frame = np.concatenate([frame] * 3, axis=-1)

        # Convert to BGR for OpenCV drawing
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Text settings
        height, width, _ = frame_bgr.shape
        font_scale = 0.45
        font_thickness = 1
        font = cv2.FONT_HERSHEY_DUPLEX
        text_color = (0, 0, 255)
        bg_color = (255, 255, 255)

        # Text content
        overlay_text = f"Step: {step_idx}"
        if reward is not None:
            overlay_text += f" | Reward: {reward:.3f}"
        if loss is not None:
            overlay_text += f" | Loss: {loss:.3f}"

        # Text size and position
        (text_w, text_h), _ = cv2.getTextSize(overlay_text, font, font_scale, font_thickness)
        text_x, text_y = 10, 30

        # Optional: draw background rectangle for better contrast
        cv2.rectangle(
            frame_bgr,
            (text_x - 5, text_y - text_h - 5),
            (text_x + text_w + 5, text_y + 5),
            bg_color,
            thickness=-1
        )

        # Draw the text
        cv2.putText(
            frame_bgr, overlay_text, (text_x, text_y),
            font, font_scale, text_color, font_thickness, cv2.LINE_AA
        )

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
