import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import hashlib
import json
from pathlib import Path

from wandb.apis.public import Runs, Run


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# TODO: Fix y axis to min/max
# TODO: Side by Side copmparison
# TODO: Normalizer
# TODO: new structure

def hash_dict(d: dict) -> str:
    h = hashlib.md5()
    h.update(json.dumps(d, sort_keys=True, cls=NumpyArrayEncoder).encode())
    return h.hexdigest()


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
api = wandb.Api()

RUN_NAME = "MT_Jan_06_22_00_Results_Halfcheetah_Test_5_full"
PARAM_DECAYS = [0.0, 0.002, 0.005, 0.007]
WORKING_DIR = "./analysis/halfcheetah/"
ENV_PARAM_CONFIG_KEY = "cheetah_decay_rate"
ENV_PARAM_LOG_KEY = "env_params/gear_scale"
ENV_PARAM_LABEL = "gear_scale" # y Axis
REPLAY_BUFFER_SIZE = [200000, 500000]

ROW_KEY = "model_perturb_rate"      # or "policy_perturb_rate"
LINE_KEY = "policy_perturb_rate"    # or "model_perturb_rate"

LINE_KEY, ROW_KEY = ROW_KEY, LINE_KEY

PALETTE_1 = ["#27448a", "#267aa9", "#3ba38c", "#f39c3f"]
PALETTE_2 = ["#762a83", "#9970ab", "#5aae61", "#1b7837"]
LINESTYLES = ["solid", "dashed", "dashdot", "dotted"]

PALETTE = PALETTE_1 if "model" in ROW_KEY else PALETTE_2

CACHE_DIR = Path(WORKING_DIR + "data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

filters = {
    "config.env_param_mode": "exponential",
    f"config.{ENV_PARAM_CONFIG_KEY}": {"$in": PARAM_DECAYS},
    "config.replay_buffer_mode": {"$in": ["reset"]},
    "config.replay_buffer_size": {"$in": REPLAY_BUFFER_SIZE},
}

cache_file = CACHE_DIR / f"{hash_dict({'name': RUN_NAME} | filters)}.pkl"


# ------------------------------------------------------------------
# Fetch runs + cache
# ------------------------------------------------------------------
if cache_file.exists():
    with open(cache_file, "rb") as f:
        env_dfs, df = pkl.load(f)
else:
    runs: Runs = api.runs(f"kiten-ethz/{RUN_NAME}", filters=filters)
    print("n runs:", len(runs))

    # ---- env parameter trajectories (one per decay)
    env_dfs = {}
    for decay in PARAM_DECAYS:
        run_for_decay: Run = next(
            r for r in runs if r.config.get(ENV_PARAM_CONFIG_KEY) == decay
        )
        env_df = run_for_decay.history(
            keys=[ENV_PARAM_LOG_KEY],
            x_axis="global_step",
            pandas=True,
        )
        env_dfs[decay] = env_df.dropna(subset=[ENV_PARAM_LOG_KEY])

    # ---- evaluation histories
    df = runs.histories(
        keys=["evaluation/average_returns"],
        x_axis="global_step",
        samples=1000,
        format="pandas",
    )

    df["policy_perturb_rate"] = df["run_id"].map(
        {r.id: r.config.get("policy_perturb_rate") for r in runs}
    )
    df["model_perturb_rate"] = df["run_id"].map(
        {r.id: r.config.get("model_perturb_rate") for r in runs}
    )
    df["decay"] = df["run_id"].map(
        {r.id: r.config.get(ENV_PARAM_CONFIG_KEY) for r in runs}
    )

    df = df.dropna()

    tmp = cache_file.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pkl.dump((env_dfs, df), f, protocol=pkl.HIGHEST_PROTOCOL)
    tmp.replace(cache_file)


# ------------------------------------------------------------------
# Aggregate
# ------------------------------------------------------------------
grouped = (
    df.groupby([ROW_KEY, LINE_KEY, "decay", "global_step"])
    .agg(
        mean=("evaluation/average_returns", "mean"),
        std=("evaluation/average_returns", "std"),
        n=("evaluation/average_returns", "count"),
    )
    .reset_index()
)
grouped["stderr"] = grouped["std"] / np.sqrt(grouped["n"])


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
row_vals = sorted(grouped[ROW_KEY].unique())
line_vals = sorted(grouped[LINE_KEY].unique())

colors = {v: PALETTE[i] for i, v in enumerate(line_vals)}
styles = {v: LINESTYLES[i] for i, v in enumerate(line_vals)}

fig, axes = plt.subplots(
    len(row_vals) + 1,
    len(PARAM_DECAYS),
    figsize=(4 * len(PARAM_DECAYS), 2.5 * (len(row_vals) + 1)),
    sharex=True,
)

# ---- Top row: env parameter per decay
for j, decay in enumerate(PARAM_DECAYS):
    ax = axes[0, j]
    env_df = env_dfs[decay]
    x_env = (
        env_df["global_step"]
        if "global_step" in env_df.columns
        else env_df.index
    )
    ax.plot(x_env, env_df[ENV_PARAM_LOG_KEY], color="black")
    ax.set_title(f"decay={decay}")
    ax.grid(True, alpha=0.3)

axes[0, 0].set_ylabel(ENV_PARAM_LABEL)

# ---- Remaining rows: perturbation plots
for i, rv in enumerate(row_vals, start=1):
    for j, decay in enumerate(PARAM_DECAYS):
        ax = axes[i, j]
        sub = grouped[
            (grouped[ROW_KEY] == rv) & (grouped["decay"] == decay)
        ]

        for lv in line_vals:
            g = sub[sub[LINE_KEY] == lv]
            ax.plot(
                g["global_step"],
                g["mean"],
                label=f"{LINE_KEY}={lv}",
                color=colors[lv],
                linestyle=styles[lv],
            )
            ax.fill_between(
                g["global_step"],
                g["mean"] - g["stderr"],
                g["mean"] + g["stderr"],
                color=colors[lv],
                alpha=0.15,
            )

        if j == 0:
            ax.set_ylabel(f"{ROW_KEY}={rv}")

        ax.grid(True, alpha=0.3)

for ax in axes[-1, :]:
    ax.set_xlabel("global_step")
axes[1, 0].legend(loc="best")

plt.tight_layout()
plt.show()

print(1)