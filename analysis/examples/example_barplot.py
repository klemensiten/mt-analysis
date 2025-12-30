import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'

# Reload the CSV to ensure the workflow starts from data loading
csv_path = "./analysis/examples/data/data_def.csv"
df_flexible = pd.read_csv(csv_path)

# Convert numeric columns
numeric_cols = ["mean", "pets", "combrl", "copax", "mean_std", "pets_std", "combrl_std", "copax_std", "base"]
df_flexible[numeric_cols] = df_flexible[numeric_cols].apply(pd.to_numeric)

# Recompute domain-task mappings for plotting
task_rows = ["primary", "secondary"]
domain_task_map = {task: [] for task in task_rows}

for task in task_rows:
    df_task = df_flexible[df_flexible["task"] == task]
    for _, row in df_task.iterrows():
        domain_label = row["domain"]
        if "Pendulum" in domain_label:
            domain_label += f" ({row['task name']})"
        if domain_label not in domain_task_map[task]:
            domain_task_map[task].append(domain_label)

n_rows = len(task_rows)
n_cols = max(len(domains) for domains in domain_task_map.values())
figure_width_inch = 6.4
aspect_ratio = 4 / 3
figure_height_inch = 3.2

# Reapply font settings
plt.rcParams.update({
    'font.size': 7.5,
    'axes.titlesize': 7.5,
    'axes.labelsize': 7.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 9,
})

# Colors (consistent)
colors = (
    '#A27AC5',   # Mean Planning
    '#3E9556',   # PETS
    'tab:red',   # COMBRL (Ours)
    '#fc9272'    # Unsupervised COMBRL (Ours)
)

# Hatching styles to improve legibility
hatches = (
    '//',   # Mean Planning → dashed hatch
    '..',   # PETS → dotted hatch
    '',     # COMBRL (Ours) → no hatch
    ''      # Unsupervised COMBRL (Ours) → no hatch
)

labels = (
    'Mean Planning',
    'PETS',
    'COMBRL (Ours)',
    'Unsupervised COMBRL (Ours)'
)

# Generate plot
fig, axes = plt.subplots(n_rows, n_cols, figsize=(figure_width_inch, figure_height_inch), squeeze=False)

for i, task in enumerate(task_rows):
    for j in range(n_cols):
        ax = axes[i, j]
        try:
            domain_label = domain_task_map[task][j]
        except IndexError:
            ax.set_visible(False)
            continue

        if "Pendulum" in domain_label:
            task_name = domain_label.split("(", 1)[1].rstrip(")")
            domain = "Pendulum"
        else:
            domain = domain_label
            task_name = None

        filtered = df_flexible[(df_flexible["domain"] == domain) & (df_flexible["task"] == task)]
        if task_name:
            filtered = filtered[filtered["task name"] == task_name]

        if filtered.empty:
            ax.set_visible(False)
            continue

        row = filtered.squeeze()
        means = row[["mean", "pets", "combrl", "copax"]].values
        stds = row[["mean_std", "pets_std", "combrl_std", "copax_std"]].values
        base = row["base"]

        if domain == "Pendulum":
            means += base

        x = np.arange(len(means))
        bars = ax.bar(
            x, means - base,
            yerr=stds,
            color=colors,
            bottom=base
        )

        # Apply hatching
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        # Title formatting with task names in parentheses
        task_title = f"({row['task name']})"
        if task == "primary":
            ax.set_title(r"$\bf{" + f"{row['domain']}" + "}$" + f"\n{task_title}")
        else:
            ax.set_title(task_title)

        ax.set_xticks([])
        ax.tick_params(axis='y', which='major', pad=0.5)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

# Row labels
axes[0][0].annotate("Primary task", xy=(-0.35, 0.5), xycoords='axes fraction',
                    ha='right', va='center', rotation=90)
axes[1][0].annotate("Downstream task", xy=(-0.35, 0.5), xycoords='axes fraction',
                    ha='right', va='center', rotation=90)

# Common labels
fig.supylabel("Average evaluation return", fontsize=9)
fig.legend(bars, labels, loc='lower center', ncol=4, frameon=False)

plt.subplots_adjust(left=0.1, bottom=0.15, right=0.99, top=0.85, wspace=0.55, hspace=0.35)
plt.show()
