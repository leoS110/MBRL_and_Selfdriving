#just a plotting script to have a live subplot of velocity next to rendered agent for demo videos, AI generated since just plotting

"""
Gymnasium wrapper: renders the env frame side-by-side with a live plot of the
cheetah's forward velocity, so RecordVideo saves both in one video.

Wrap it INSIDE RecordVideo (RecordVideo calls render() on whatever it wraps):

    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
    env = VelocityPlotWrapper(env, max_steps=video_length)
    env = gym.wrappers.RecordVideo(env, video_folder=..., video_length=video_length)

Plotted signal: info["x_velocity"] = (x_after - x_before) / dt, i.e. the torso
(root) velocity that HalfCheetah's forward reward is computed from.
"""

import numpy as np
import gymnasium as gym
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg  # off-screen, no GUI window


class VelocityPlotWrapper(gym.Wrapper):
    def __init__(self, env, max_steps=500, v_lim=(-3.0, 10.0), plot_width=480, target_velocity=None, dpi=100):
        super().__init__(env)
        assert env.render_mode == "rgb_array", "VelocityPlotWrapper needs render_mode='rgb_array'"

        self.dt = env.unwrapped.dt  # seconds per env step = model timestep * frame_skip
        self.times = []
        self.velocities = []

        # --- build the figure once; each frame only updates the line data ---
        height = env.unwrapped.height  # MuJoCo frame height in pixels (480 by default)
        self.fig = Figure(figsize=(plot_width / dpi, height / dpi), dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)

        ax = self.fig.add_subplot()
        ax.set_xlim(0, max_steps * self.dt)  # fixed axes: line grows left to right, no rescaling jitter
        ax.set_ylim(*v_lim)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("forward velocity [m/s]")
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.grid(alpha=0.3)

        (self.line,) = ax.plot([], [], linewidth=1.5)
        if target_velocity is not None:
            ax.axhline(target_velocity, color="tab:red", linestyle="--", linewidth=1.2,
                label=f"target = {target_velocity:.1f} m/s")
            ax.legend(loc="upper right", fontsize=9)  # upper left is taken by the readout

        self.readout = ax.text(0.03, 0.97, "", transform=ax.transAxes,
                               va="top", family="monospace", fontsize=9)
        self.fig.tight_layout()

    def reset(self, **kwargs):
        self.times.clear()
        self.velocities.clear()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.times.append((len(self.times) + 1) * self.dt)
        self.velocities.append(info["x_velocity"])
        return obs, reward, terminated, truncated, info

    def render(self):
        frame = self.env.render()  # (H, W, 3) uint8 from MuJoCo

        self.line.set_data(self.times, self.velocities)
        if self.velocities:
            self.readout.set_text(f"v    = {self.velocities[-1]:5.2f} m/s\n"
                                  f"mean = {np.mean(self.velocities):5.2f} m/s")
        else:
            self.readout.set_text("")

        self.canvas.draw()
        plot = np.asarray(self.canvas.buffer_rgba())[..., :3]  # drop alpha -> (H, plot_width, 3)

        return np.hstack([frame, plot])