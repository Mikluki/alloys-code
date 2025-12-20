from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dpi = 350
style_path = "/home/mik/templates/plt/article-f1-12x6.mplstyle"
save_ext = ".pdf"
YLIM_MIN = None
YLIM_MAX = None
SHOW = False

csv_path = Path("cpu-ncore-kpar-rand32/experiments.csv")

calc_type = csv_path.name.split("-")[-1]
calc_type_path = Path("x-" + calc_type)
calc_type_path.mkdir(exist_ok=True, parents=True)

# Parse the CSV data
df = pd.read_csv(csv_path, sep=",", comment="#")

print(f"Total experiments in CSV: {len(df)}")

# Filter for completed jobs only
completed_df = df[df["status"] == "COMPLETED"].copy()
print(f"Completed experiments: {len(completed_df)}")

if len(completed_df) == 0:
    print("No completed experiments found! Run collect_results.py first.")
    exit(1)

# Extract directory name from path using pathlib (last component without trailing underscore)
completed_df["dir_name"] = completed_df["directory"].apply(  # pyright: ignore
    lambda x: Path(x).name.rstrip("_")
)


# Convert elapsed_time (string from VASP) to float seconds, then to minutes
def convert_elapsed_time(time_str):
    """Convert VASP elapsed time string to float seconds."""
    try:
        return float(time_str) if time_str else np.nan
    except (ValueError, TypeError):
        return np.nan


completed_df["runtime_seconds"] = completed_df["elapsed_time"].apply(  # pyright: ignore
    convert_elapsed_time
)
completed_df["runtime_minutes"] = completed_df["runtime_seconds"] / 60

# Remove rows where elapsed_time conversion failed
completed_df = completed_df.dropna(subset=["runtime_seconds"])  # pyright: ignore
print(f"Valid timing data: {len(completed_df)}")

if len(completed_df) == 0:
    print("No valid timing data found!")
    exit(1)

# Ensure ncore and kpar are integers
completed_df["ncore"] = completed_df["ncore"].astype(int)
completed_df["kpar"] = completed_df["kpar"].astype(int)

# Create a unique identifier for each ncore-kpar combination
completed_df["config"] = completed_df.apply(
    lambda row: f"ncore={row['ncore']}, kpar={row['kpar']}", axis=1
)

# Get unique ncore-kpar configurations
unique_configs = sorted(completed_df["config"].unique())
print(f"Unique configurations: {unique_configs}")

# Get unique ntasks values for x-axis ticks
unique_ntasks = sorted(completed_df["ntasks"].unique())
print(f"Unique ntasks values: {unique_ntasks}")

# Get unique directories
unique_dirs = sorted(completed_df["dir_name"].unique())
print(f"Unique directories: {len(unique_dirs)}")

# Check if style file exists, if not use default
if Path(style_path).exists():
    plt.style.use(style_path)
else:
    print(f"Style file {style_path} not found, using default style")

# 1. First create a combined plot with all data
plt.figure(figsize=(10, 6))

# Use viridis colormap for different ncore-kpar combinations
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_configs)))  # pyright: ignore

# Plot each ncore-kpar configuration with a different color
for i, config in enumerate(unique_configs):
    subset = completed_df[completed_df["config"] == config]
    if not subset.empty:
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
plt.tight_layout()

fname_all = calc_type_path / f"00-bench-all-dirs{save_ext}"
plt.savefig(fname_all, dpi=dpi)
print(f"Saved combined plot: {fname_all}")

# 2. Now create separate plots for each unique directory
for dir_name in unique_dirs:
    # Filter data for this directory
    dir_df = completed_df[completed_df["dir_name"] == dir_name]

    if dir_df.empty:
        print(f"No data for directory {dir_name}, skipping...")
        continue

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot each ncore-kpar configuration with a different color
    configs_in_dir = []
    for i, config in enumerate(unique_configs):
        subset = dir_df[dir_df["config"] == config]
        if not subset.empty:  # pyright: ignore
            plt.scatter(
                subset["ntasks"],
                subset["runtime_minutes"],
                label=config,
                color=colors[i],
                s=100,
                alpha=0.8,
            )
            configs_in_dir.append(config)

    # Set x-ticks to the unique ntasks values from the entire dataset
    plt.xticks(unique_ntasks, labels=unique_ntasks)

    # Add labels and title
    plt.xlabel("Number of CPUs (ntasks)")
    plt.ylabel("Runtime (minutes)")
    plt.title(f"Runtime vs n-CPUs for {dir_name}")

    plt.ylim(YLIM_MIN, YLIM_MAX)
    plt.grid(True, linestyle="--", alpha=0.3)

    # Only add a legend if we have more than one configuration for this directory
    if len(configs_in_dir) > 1:
        plt.legend(title="Configuration", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        plt.annotate(
            f"Configuration: {configs_in_dir[0] if configs_in_dir else 'Unknown'}",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    plt.tight_layout()

    fname_dir = calc_type_path / f"01-bench-dir-{dir_name}{save_ext}"
    plt.savefig(fname_dir, dpi=dpi)
    print(f"Saved directory plot: {fname_dir}")

# Display all plots
if SHOW:
    plt.show()

# Print summary of the processed data
print("\n=== DATA SUMMARY ===")
print(f"Experiments by status:")
status_counts = df["status"].value_counts()
for status, count in status_counts.items():
    print(f"  {status}: {count}")

print(f"\nCompleted experiments runtime summary:")
print(
    completed_df[
        ["ntasks", "runtime_minutes", "ncore", "kpar", "config", "dir_name"]
    ].describe()
)

print(f"\nSample data:")
print(
    completed_df[
        ["ntasks", "runtime_minutes", "ncore", "kpar", "config", "dir_name"]
    ].head()
)

# Summary by configuration
print(f"\nRuntime summary by configuration:")
config_summary = (
    completed_df.groupby("config")["runtime_minutes"]
    .agg(["count", "mean", "std"])
    .round(2)
)
print(config_summary)
