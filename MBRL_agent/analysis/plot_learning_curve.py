#fully claude plotting of logs recorded during MBRL_train.py


import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator, StrMethodFormatter

default_csv = Path(__file__).resolve().parent / "progress.csv"
csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_csv
log = pd.read_csv(csv_path)

aggregation = log["time/iteration"]
reward = log["rollout/mean_step_reward"]
random_reward = log["baseline/random_step_reward"].dropna().iloc[0]
loss_first = log["train/loss_first"]  # mean loss over the first epoch of each retrain
loss_last = log["train/loss_last"]    # mean loss over the last epoch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "pdf.fonttype": 42,  # embed fonts as TrueType so text stays editable
})

BLUE = "#2a78d6"
LIGHT_BLUE = "#86b6ef"
GREY = "#9a9a9a"
style = dict(lw=1.5, ms=4.5, mec="white", mew=0.8)

fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(6.5, 2.4), layout="constrained")

# (a) reward per step of the MPC controller, with the random-action baseline
ax_r.axhline(random_reward, color=GREY, lw=1, ls=(0, (4, 3)))
ax_r.annotate("random actions", xy=(aggregation.iloc[-1], random_reward),
              xytext=(0, 2), textcoords="offset points",
              ha="right", va="bottom", color="#555555",
              bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
ax_r.plot(aggregation, reward, "-o", color=BLUE, **style)
ax_r.set_ylabel("Mean reward per environment step")

# (b) dynamics-model loss. aggregation 0 trains from random weights (loss ~1),
# later aggregations are warm-started, so the first-epoch loss is only
# comparable from aggregation 1 onwards.
ax_l.plot(aggregation, loss_first, "-o", color=LIGHT_BLUE,
          label="Start", **style)
#ax_l.plot(aggregation.iloc[1:], loss_first.iloc[1:], "-o", color=LIGHT_BLUE,
          #label="First epoch", **style)
ax_l.plot(aggregation, loss_last, "-o", color=BLUE, label="End", **style)
ax_l.set_yscale("log")
ax_l.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
ax_l.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
ax_l.yaxis.set_minor_locator(NullLocator())
ax_l.set_ylabel("Mean training loss across ensemble")
ax_l.legend(loc="upper right", handlelength=1.8)

for ax, tag in [(ax_r, "(a)"), (ax_l, "(b)")]:
    ax.set_xlabel("aggregation")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title(tag, loc="left", fontsize=9)

out = csv_path.with_name("training_progress")
fig.savefig(out.with_suffix(".pdf"))
fig.savefig(out.with_suffix(".png"), dpi=300)