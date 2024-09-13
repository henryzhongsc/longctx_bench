import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Create a dummy heatmap
data = np.random.rand(10, 10)
cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
ax = sns.heatmap(data, cmap=cmap, cbar=True, vmin=0, vmax=1)

# Extract the colorbar from the heatmap
cbar = ax.collections[0].colorbar

# Create a new figure and axis for the colorbar
fig, ax = plt.subplots(figsize=(2, 5))

# Create a new colorbar in the new figure
bar = fig.colorbar(cbar.mappable)

# Change the font size of the units here
bar.ax.tick_params(labelsize=12)

# Remove axis ticks and labels
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# plt.show()
plt.savefig('./bar.pdf', bbox_inches='tight', pad_inches=0.0)