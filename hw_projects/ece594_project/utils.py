import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_matrix_as_heatmap_with_highlight(df, save_path, x_label, y_label):
    sns.set_context('talk')
    plt.figure(figsize=(10, 8))
    
    # Creating the heatmap
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap='viridis', linewidths=.5, cbar_kws={'label': 'Scale'})
    
    # Customizing the plot with the provided labels
    plt.title('Matrix Visualization')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Find the maximum value's location
    max_val = df.max().max()
    max_pos = df.stack().idxmax()
    
    # Convert the DataFrame index/col to heatmap axis position
    y, x = df.index.get_loc(max_pos[0]), df.columns.get_loc(max_pos[1])
    
    # Add a circle around the cell with the maximum value
    circle = patches.Circle((x, y), 0.5, linewidth=2, edgecolor='red', facecolor='none', clip_on=False)
    ax.add_patch(circle)
    
    # Adjust the layout
    plt.tight_layout()
    
    # Check if the directory exists, if not, create it
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the figure to the specified path
    plt.savefig(save_path)
    
    # Close the plot figure to free memory
    plt.close()