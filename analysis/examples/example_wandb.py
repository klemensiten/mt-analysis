import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

import hashlib
import json
import numpy as np

from wandb.apis.public import Run, Runs
from pandas import DataFrame


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


api = wandb.Api()

MODEL_PERTURB_RATES = [0.0, 0.2, 0.5, 1.0]
CACHED = False
RUN_NAME = "MT_Dez_23_16_30_Resets_Test_1"

filters = {
    "config.env_param_mode": "episodic",
    "config.replay_buffer_mode": {"$in": ["reset"]},
    "config.replay_buffer_size": {"$in": [2000, 6000]},
    "config.model_perturb_rate": {"$in": MODEL_PERTURB_RATES},
}

runs: Runs = api.runs(
    f"kiten-ethz/{RUN_NAME}",
    filters=filters,
)

print("n runs:", len(runs))

# -------------------------------------------------
# 1) Fetch env param trajectory (take first run)
# -------------------------------------------------
if not CACHED:
    ref_run: Run = runs[0]

    env_df = ref_run.history(
        keys=["env_params/max_torque"],
        x_axis="global_step",
        pandas=True,
    )

    env_df = env_df.dropna(subset=["env_params/max_torque"])

    # -------------------------------------------------
    # 2) Fetch evaluation histories for all runs
    # -------------------------------------------------
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

    df = df.dropna(subset=[
        "evaluation/average_returns",
        "policy_perturb_rate",
        "model_perturb_rate",
    ])

    file_name = hash_dict({'name': RUN_NAME} | filters)
    with open(f'./analysis/examples/data/{file_name}.pkl', 'wb') as file:
        pkl.dump((env_df, df), file, protocol=pkl.HIGHEST_PROTOCOL)

else:
    file_name = hash_dict({'name': RUN_NAME} | filters)
    with open(f'./analysis/examples/data/{file_name}.pkl', 'rb') as file:
        env_df, df = pkl.load(file)
    

# -------------------------------------------------
# 3) Aggregate: mean + stderr
# -------------------------------------------------
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

# -------------------------------------------------
# 4) Plot
# -------------------------------------------------
n_rows = 1 + len(MODEL_PERTURB_RATES)

fig, axes = plt.subplots(
    nrows=n_rows,
    ncols=1,
    figsize=(8, 3 * n_rows),
    sharex=True,
)

# ---- Top: env param plot
ax_env = axes[0]
ax_env.plot(
    env_df["global_step"] if "global_step" in env_df else env_df.index,
    env_df["env_params/max_torque"],
)
ax_env.set_title("Environment parameter: max torque")
ax_env.set_ylabel("max_torque")
ax_env.grid(True, alpha=0.3)

# ---- Below: one row per model_perturb_rate
for ax, m_rate in zip(axes[1:], MODEL_PERTURB_RATES):
    sub = grouped[grouped["model_perturb_rate"] == m_rate]

    for p_rate, g in sub.groupby("policy_perturb_rate"):
        ax.plot(
            g["global_step"],
            g["mean_return"],
            label=f"policy={p_rate}",
        )
        ax.fill_between(
            g["global_step"],
            g["mean_return"] - g["stderr"],
            g["mean_return"] + g["stderr"],
            alpha=0.2,
        )

    ax.set_title(f"model_perturb_rate = {m_rate}")
    ax.set_ylabel("avg eval return")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("global_step")
axes[1].legend(loc="best")

plt.tight_layout()
plt.show()

print(1)