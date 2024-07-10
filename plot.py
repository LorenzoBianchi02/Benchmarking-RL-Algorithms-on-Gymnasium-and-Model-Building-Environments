import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import sys
import numpy as np

def plot_algorithms(main_dir, max_timesteps=None, interval=250, time_col='time/time_elapsed', reward_col='rollout/ep_rew_mean', total_timesteps_col='time/total_timesteps'):
    """
    Plots the mean episode reward over total timesteps for each algorithm with mean and standard deviation shaded region.
    
    Args:
        main_dir (str): The main directory containing subdirectories for each algorithm.
        max_timesteps (int, optional): The maximum number of timesteps to include in the plot. Defaults to None.
        interval (int, optional): The interval for grouping time elapsed values. Defaults to 100.
        time_col (str, optional): The column name for time in the CSV files. Defaults to 'time/time_elapsed'.
        reward_col (str, optional): The column name for rewards in the CSV files. Defaults to 'rollout/ep_rew_mean'.
        total_timesteps_col (str, optional): The column name for total timesteps in the CSV files. Defaults to 'time/total_timesteps'.
    """
    algorithms = os.listdir(main_dir)
    
    # Set the style and color palette
    sns.set(style="whitegrid")
    colors = sns.color_palette("deep", len(algorithms))  # Get distinct colors for each algorithm
    
    plt.figure(figsize=(10, 6))
    
    for idx, algorithm in enumerate(algorithms):
        algorithm_dir = os.path.join(main_dir, algorithm)
        
        # Replace 'progress*.csv' with your specific CSV file pattern
        csv_files = glob.glob(os.path.join(algorithm_dir, 'progress*.csv'))
        
        # Load all CSV files into a list of dataframes
        dataframes = [pd.read_csv(file) for file in csv_files]
        
        # Concatenate all dataframes into a single dataframe
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Extract only the relevant columns
        combined_df = combined_df[[time_col, reward_col, total_timesteps_col]]
        
        # Filter based on max_timesteps if provided
        if max_timesteps:
            combined_df = combined_df[combined_df[time_col] <= max_timesteps]
        
        # Group by intervals of 'time_col' and calculate mean and std
        grouped = pd.cut(combined_df[time_col], bins=np.arange(0, combined_df[time_col].max() + interval, interval))
        grouped_df = combined_df.groupby(grouped)[reward_col].agg(['mean', 'std'])
        
        # Reset index to get intervals as a column
        grouped_df = grouped_df.reset_index()
        
        # Calculate the midpoint of each interval for plotting
        grouped_df['interval_midpoint'] = grouped_df[time_col].apply(lambda x: x.mid)
        
        # Interpolate missing values within each interval
        grouped_df['mean'] = grouped_df['mean'].interpolate()
        grouped_df['std'] = grouped_df['std'].interpolate()
        
        # Sort by 'interval_midpoint' to ensure lines are connected
        grouped_df = grouped_df.sort_values(by='interval_midpoint')
        
        # Get total timesteps for the algorithm
        total_timesteps = combined_df[total_timesteps_col].max()
        
        # Plot mean line with shaded region for standard deviation
        plt.plot(grouped_df['interval_midpoint'], grouped_df['mean'], color=colors[idx], label=f'{algorithm} - {round(total_timesteps)} total timesteps', linestyle='-', linewidth=1.5)
        plt.fill_between(grouped_df['interval_midpoint'], grouped_df['mean'] - grouped_df['std'], grouped_df['mean'] + grouped_df['std'], color=colors[idx], alpha=0.3)
    
    # Set labels and title
    plt.xlabel(time_col)
    plt.ylabel(reward_col)
    plt.title(f'{main_dir[8:-1]} (Interval of {interval})')
    
    # Add legend
    plt.legend()
    
    # Show plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python script_name.py <main_directory_path> [max_timesteps]")
        sys.exit(1)
    
    main_directory = sys.argv[1]
    max_timesteps = int(sys.argv[2]) if len(sys.argv) == 3 else None
    
    plot_algorithms(main_directory, max_timesteps)
