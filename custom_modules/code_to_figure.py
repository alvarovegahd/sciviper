import numpy as np
import matplotlib.pyplot as plt

# Sample data (replace with actual data if available)
sessions = np.arange(1, 10)

data = {
    'Ft-CNN': [60, 25, 15, 8, 4, 3, 3, 3, 3],
    'iCaRL*': [60, 45, 35, 25, 20, 18, 17, 16, 15],
    'EEIL*': [60, 40, 30, 22, 18, 16, 15, 14, 13],
    'NCM*': [60, 42, 32, 24, 19, 17, 16, 15, 14],
    'Ours-AL': [60, 48, 38, 28, 22, 19, 17, 16, 15],
    'Ours-AL-MML': [60, 46, 36, 26, 20, 18, 16, 15, 14],
    'Joint-CNN': [60, 55, 45, 35, 25, 18, 15, 13, 12]
}

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Plot each line with distinct colors, markers, and line styles
ax.plot(sessions, data['Ft-CNN'], marker='o', linestyle='-', color='blue', label='Ft-CNN')
ax.plot(sessions, data['iCaRL*'], marker='s', linestyle='-', color='cyan', label='iCaRL*')
ax.plot(sessions, data['EEIL*'], marker='^', linestyle='-', color='orange', label='EEIL*')
ax.plot(sessions, data['NCM*'], marker='D', linestyle='-', color='yellow', label='NCM*')
ax.plot(sessions, data['Ours-AL'], marker='v', linestyle='-', color='green', label='Ours-AL')
ax.plot(sessions, data['Ours-AL-MML'], marker='*', linestyle='-', color='purple', label='Ours-AL-MML')
ax.plot(sessions, data['Joint-CNN'], marker='x', linestyle='-', color='black', label='Joint-CNN')

# Set axis labels, title, and grid
ax.set_xlabel('Sessions')
ax.set_ylabel('Overall Acc.(%)')
ax.set_title('(a) 5-way 10-shot')
ax.grid(True)

# Add legend
ax.legend()

# Adjust figure size and resolution
plt.tight_layout()
plt.savefig('custom_modules/images/reconstructed_plot.png', dpi=300)

# Show the plot
plt.show()

plt.savefig("custom_modules/images/charxiv_0_qwen2.5-VL-72B.png")