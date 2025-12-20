from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dpi = 350
style_path = "/home/mik/templates/plt/article-f1-12x6.mplstyle"
save_ext = ".pdf"
YLIM_MIN = 0
YLIM_MAX = 55
SHOW = False

csv_path = Path("filtered-data/all-exp-rand.csv")
# csv_path = Path("filtered-data/all-exp-cats.csv")

calc_type = csv_path.name.split("-")[-1]
calc_type_path = Path("x-" + calc_type)
calc_type_path.mkdir(exist_ok=True, parents=True)

# Parse the CSV data
df = pd.read_csv(csv_path, sep=",", comment="#")

# Extract directory name from path using pathlib (last component without trailing underscore)
df["dir_name"] = df["directory"].apply(lambda x: Path(x).name.rstrip("_"))

# Fill empty ncore and kpar values with 1 as specified
df["ncore"] = df["ncore"].fillna(1).astype(int)
df["kpar"] = df["kpar"].fillna(1).astype(int)

# Convert runtime from seconds to minutes
df["runtime_minutes"] = df["runtime_seconds"] / 60

# Create a unique identifier for each ncore-kpar combination
df["config"] = df.apply(lambda row: f"ncore={row['ncore']}, kpar={row['kpar']}", axis=1)

# Get unique ncore-kpar configurations
unique_configs = df["config"].unique()

# Get unique ntasks values for x-axis ticks
unique_ntasks = sorted(df["ntasks"].unique())

# Get unique directories
unique_dirs = df["dir_name"].unique()

# Plt Style
plt.style.use(style_path)

# 1. First create a combined plot with all data
plt.figure(figsize=(10, 6))

# Use viridis colormap for different ncore-kpar combinations
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_configs)))  # pyright: ignore

# Plot each ncore-kpar configuration with a different color
for i, config in enumerate(unique_configs):
    subset = df[df["config"] == config]
    plt.scatter(
        subset["ntasks"],
        subset["runtime_minutes"],
        label=config,
        color=colors[i],
        s=100,
        alpha=0.8,
    )

# Set x-ticks to the unique ntasks values
plt.xticks(unique_ntasks, labels=unique_ntasks)

# Add labels and title
plt.xlabel("Number of CPUs (ntasks)")
plt.ylabel("Runtime (minutes)")
plt.title("Runtime vs n-CPUs for all Structures")
plt.ylim(YLIM_MIN, YLIM_MAX)
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(title="Configuration", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.legend(title="Configuration")
plt.tight_layout()

fname_all = Path(calc_type_path, f"00-bench-all-dirs_{save_ext}")
plt.savefig(fname_all, dpi=dpi)

# # Add directory name annotations to each point
# for _, row in df.iterrows():
#     plt.annotate(
#         row["dir_name"],  # pyright: ignore
#         (row["ntasks"], row["runtime_minutes"]),  # pyright: ignore
#         textcoords="offset points",
#         xytext=(0, 7),
#         ha="center",
#         fontsize=8,
#     )


# 2. Now create separate plots for each unique directory
for dir_name in unique_dirs:
    # Filter data for this directory
    dir_df = df[df["dir_name"] == dir_name]

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot each ncore-kpar configuration with a different color
    for i, config in enumerate(unique_configs):
        subset = dir_df[dir_df["config"] == config]
        if (
            not subset.empty  # pyright: ignore
        ):  # Only plot if there's data for this config    # pyright: ignore
            plt.scatter(
                subset["ntasks"],
                subset["runtime_minutes"],
                label=config,
                color=colors[i],
                s=100,
                alpha=0.8,
            )

    # Set x-ticks to the unique ntasks values from the entire dataset
    plt.xticks(unique_ntasks, labels=unique_ntasks)

    # Add labels and title
    plt.xlabel("Number of CPUs (ntasks)")
    plt.ylabel("Runtime (minutes)")
    plt.title(f"Runtime vs n-CPUs for {dir_name}")

    plt.ylim(YLIM_MIN, YLIM_MAX)
    plt.grid(True, linestyle="--", alpha=0.3)

    # Only add a legend if we have more than one configuration for this directory
    if len(dir_df["config"].unique()) > 1:  # pyright: ignore
        plt.legend(title="Configuration", bbox_to_anchor=(1.05, 1), loc="upper left")
        # plt.legend(title="Configuration")
    else:
        plt.annotate(
            f"Configuration: {dir_df['config'].iloc[0]}",  # pyright: ignore
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    plt.tight_layout()

    fname_dir = Path(calc_type_path, f"01-bench-dir-{dir_name}_{save_ext}")
    plt.savefig(fname_dir, dpi=dpi)

# Display all plots
if SHOW is True:
    plt.show()

# Print summary of the processed data
print(df[["ntasks", "runtime_minutes", "ncore", "kpar", "config", "dir_name"]].head())
