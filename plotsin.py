import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import sys

def plot_algorithms(main_dir, max_timesteps=None):
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
        combined_df = combined_df[['time/time_elapsed', 'cumulative_reward']]

        if max_timesteps:
            combined_df = combined_df[combined_df['episode_num'] <= max_timesteps]
        
        # Group by 'episode_num' and calculate mean and std
        grouped_df = combined_df.groupby('episode_num')['cumulative_reward'].agg(['mean', 'std']).reset_index()
        
        # Plot mean line with shaded region for standard deviation
        sns.lineplot(data=grouped_df, x='episode_num', y='mean', color=colors[idx], label=f'{algorithm} Mean')
        plt.fill_between(grouped_df['episode_num'], grouped_df['mean'] - grouped_df['std'], grouped_df['mean'] + grouped_df['std'], color=colors[idx], alpha=0.3)
    
    # Set labels and title
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title(main_dir[8:-1])
    
    # Add legend
    plt.legend()
    
    # Show plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main_directory = sys.argv[1]
    max_timesteps = int(sys.argv[2]) if len(sys.argv) == 3 else None
    plot_algorithms(main_directory, max_timesteps)
