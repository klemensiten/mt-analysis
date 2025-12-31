import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

import hashlib
import json
from pathlib import Path
from matplotlib.colors import Normalize

from wandb.apis.public import Run, Runs


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def hash_dict(d: dict) -> str:
    dhash = hashlib.md5()
    dhash.update(json.dumps(d, sort_keys=True, cls=NumpyArrayEncoder, indent=4).encode())
    return dhash.hexdigest()


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
api = wandb.Api()

MODEL_PERTURB_RATES = [0.0, 0.2, 0.5, 1.0]
CACHED = True
RUN_NAME = "MT_Dez_30_13_40_Exponential_Test_2"
ROW_KEY = "model_perturb_rate"
LINE_KEY = "policy_perturb_rate"

filters = {
    "config.env_param_mode": "exponential",
    "config.pendulum_torque_decay_rate": {"$in": [0.05]},
    "config.replay_buffer_mode": {"$in": ["reset"]},
    "config.replay_buffer_size": {"$in": [2000, 6000]},
    "config.model_perturb_rate": {"$in": MODEL_PERTURB_RATES},
}

CACHE_DIR = Path("./analysis/examples/data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Fetch runs
# ------------------------------------------------------------------
runs: Runs = api.runs(f"kiten-ethz/{RUN_NAME}", filters=filters)
print("n runs:", len(runs))

file_name = hash_dict({"name": RUN_NAME} | filters)
cache_file = CACHE_DIR / f"{file_name}.pkl"

if CACHED and cache_file.exists():
    with open(cache_file, "rb") as f:
        env_df, df = pkl.load(f)
else:
    ref_run: Run = runs[0]

    # Env parameters (same for all runs)
    env_df = ref_run.history(
            keys=["env_params/max_torque"],
            x_axis="global_step",
            pandas=True,
        )

    # Evaluation metrics
    df = runs.histories(
        keys=["evaluation/average_returns"],
        x_axis="global_step",
        samples=1000,
        format="pandas",
    )

    policy_rates = {r.id: r.config.get("policy_perturb_rate") for r in runs}
    model_rates = {r.id: r.config.get("model_perturb_rate") for r in runs}

    df["policy_perturb_rate"] = df["run_id"].map(policy_rates)
    df["model_perturb_rate"] = df["run_id"].map(model_rates)

    df = df.dropna(
        subset=[
            "evaluation/average_returns",
            "policy_perturb_rate",
            "model_perturb_rate",
        ]
    )

    # cache write
    tmp = cache_file.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pkl.dump((env_df, df), f, protocol=pkl.HIGHEST_PROTOCOL)
    tmp.replace(cache_file)


# ------------------------------------------------------------------
# Aggregate mean and stderr
# ------------------------------------------------------------------
grouped = (
    df.groupby(["model_perturb_rate", "policy_perturb_rate", "global_step"])
    .agg(
        mean_return=("evaluation/average_returns", "mean"),
        std_return=("evaluation/average_returns", "std"),
        n=("evaluation/average_returns", "count"),
    )
    .reset_index()
)

grouped["stderr"] = grouped["std_return"] / np.sqrt(grouped["n"])


# ------------------------------------------------------------------
# Colormap (SUMMER)
# ------------------------------------------------------------------
policy_rates_sorted = sorted(grouped["policy_perturb_rate"].unique())

cmap = plt.get_cmap("summer")
norm = Normalize(
    vmin=min(policy_rates_sorted),
    vmax=max(policy_rates_sorted),
)

colors = {p: cmap(norm(p)*0.7) for p in policy_rates_sorted}

LINESTYLES = {
    "solid": (0, ()),
    "dashed": (0, (3, 2)),
    "dashdot": (0, (4, 1, 1, 1)),
    "dotted": (0, (1, 1)),
}

linestyles = {
    p: list(LINESTYLES.values())[i % len(LINESTYLES)]
    for i, p in enumerate(policy_rates_sorted)
}

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
n_rows = 1 + len(MODEL_PERTURB_RATES)

fig, axes = plt.subplots(
    nrows=n_rows,
    ncols=1,
    figsize=(8, 3 * n_rows),
    sharex=True,
)

# ---- Top: environment parameter
x_env = (
    env_df["global_step"]
    if "global_step" in env_df.columns
    else env_df["_step"]
)

ax_env = axes[0]
ax_env.plot(x_env, env_df["env_params/max_torque"], color="black")
ax_env.set_title("Environment parameter: max torque")
ax_env.set_ylabel("max_torque")
ax_env.grid(True, alpha=0.3)

# ---- One row per model_perturb_rate
for ax, m_rate in zip(axes[1:], MODEL_PERTURB_RATES):
    sub = grouped[grouped["model_perturb_rate"] == m_rate]

    for p_rate, g in sub.groupby("policy_perturb_rate"):
        color = colors[p_rate]

        ax.plot(
            g["global_step"],
            g["mean_return"],
            label=f"policy={p_rate}",
            color=color,
            linestyle=linestyles[p_rate],
        )
        ax.fill_between(
            g["global_step"],
            g["mean_return"] - g["stderr"],
            g["mean_return"] + g["stderr"],
            color=color,
            alpha=0.15,
        )

    ax.set_title(f"model_perturb_rate = {m_rate}")
    ax.set_ylabel("avg eval return")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("global_step")
axes[1].legend(loc="best")

plt.tight_layout()
plt.show()
