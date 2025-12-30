import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams.update({'font.size': 5})

# ---------------------------------------------------------------------
# Environment setup with x-axis ranges
# ---------------------------------------------------------------------
envs = [
    ("pendulum",    "Gym-Pendulum: Swing-up",        15_000),
    ("mountaincar", "Gym-MountainCar: Go up",  25_000),    
    ("quadruped",   "DMC-Quadruped: Run",    2_000_000),
    ("cartpole",    "DMC-CartPole: Swing-up", 500_000),
    ("hstand",     "DMC-Humanoid: Stand",     2_000_000),
    ("halfcheetah", "Gym-HalfCheetah: Run",  4_000_000),
    ("hopper",      "DMC-Hopper: Hop forward",       1_000_000),
    ("reacher",     "Gym-Reacher: Reach target",     150_000),
    ("pusher",      "Gym-Pusher: Push to target",    150_000),
    ("hrun",      "DMC-Humanoid: Run",    2_000_000),
]

algs = ["COMBRL", "PETS", "Mean"]
alg_labels = {"COMBRL": "COMBRL (Ours)", "PETS": "PETS", "Mean": "Mean Planning"}

# CSV naming convention: {env}_{alg}.csv in ./analysis/ folder
path = "./analysis/examples/data/"
csv_paths = {
    (env, alg): f"{path}{env}_{alg}.csv"
    for env, _, _ in envs
    for alg in algs
}

# ---------------------------------------------------------------------
# Colors and line styles
# ---------------------------------------------------------------------
alg_colors = {
    "COMBRL": "tab:red",     # solid red
    "PETS": "#3E9556",       # green
    "Mean": "#A27AC5",       # purple
}

alg_styles = {
    "COMBRL": "solid",
    "PETS": "dashdot",
    "Mean": "dashed",
}

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
dfs = {}
steps_by_env = {}
for (env, alg), csv_path in csv_paths.items():
    df = pd.read_csv(csv_path)
    df = df.apply(pd.to_numeric)
    if "global_step" in df.columns:
        steps = df["global_step"].to_numpy()
    elif "Step" in df.columns:
        steps = df["Step"].to_numpy()
    else:
        raise ValueError(f"No step column found in {csv_path}")

    steps_by_env[env] = steps
    cols = [
        c for c in df.columns
        if c not in ["Step", "episode_idx"]
        and not c.endswith("__MIN")
        and not c.endswith("__MAX")
        and not c.endswith("_step")
    ]
    dfs[(env, alg)] = df[cols]

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 5, sharey=False, figsize=(7, 2))
axes = axes.flatten()

for ax, (env, title, max_steps) in zip(axes, envs):
    y_by_alg = {}
    se_by_alg = {}

    for alg in algs:
        key = (env, alg)
        if key not in dfs:
            continue
        df = dfs[key]
        df_num = df.select_dtypes(include=[np.number])
        if df_num.empty:
            continue

        # Mean + SE across runs
        y_mean = df_num.mean(axis=1).to_numpy()
        y_se   = df_num.sem(axis=1).to_numpy()

        y_by_alg[alg] = y_mean
        se_by_alg[alg] = y_se

    if not y_by_alg:
        ax.set_title(f"{title}\n(no data)")
        ax.grid(True)
        continue

    # Align lengths
    min_len = min(len(v) for v in y_by_alg.values())
    for alg in y_by_alg:
        y_by_alg[alg] = y_by_alg[alg][:min_len]
        se_by_alg[alg] = se_by_alg[alg][:min_len]

    steps = steps_by_env[env][:min_len]

    # Plot raw values (no normalization)
    for alg, y in y_by_alg.items():
        se = se_by_alg[alg]
        ax.plot(
            steps, y,
            label=alg,
            linewidth=1.25,
            color=alg_colors[alg],
            linestyle=alg_styles[alg]
        )
        ax.fill_between(
            steps, y - se, y + se,
            color=alg_colors[alg], alpha=0.3, linewidth=0
        )

    ax.set_title(title, fontsize=7.5)
    # ax.set_xlabel("Environment steps")
    if env=="pendulum":
        ax.set_xlim(1000, max_steps)
    else:
        ax.set_xlim(0, max_steps)

    if env=="hstand":
        ax.set_ylim(0,800)
    elif env=="hrun":
        ax.set_ylim(0,200)
    
    ax.grid(True)

    # ---- scientific notation for x-axis ----
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    ax.xaxis.set_major_formatter(formatter)

    ax.grid(True)

# Shared y-label
fig.supylabel("Average evaluation return", ha="center", va="center", fontsize=7.5)
fig.supxlabel("Environment steps", ha="center", va="center", fontsize=7.5)

# Legend
fig.legend(
    handles=[
        plt.Line2D([], [], color=alg_colors[a], linestyle=alg_styles[a], linewidth=1.25, label=alg_labels[a])
        for a in algs
    ],
    loc="upper center",
    ncol=3,
    frameon=False,
    fontsize=7.5
)

plt.subplots_adjust(left=0.08, bottom=0.12, right=0.99, top=0.82, wspace=0.3, hspace=0.65)
plt.show()
