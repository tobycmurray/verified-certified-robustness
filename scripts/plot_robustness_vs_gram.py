import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 5:
    print(f"Usage: {sys.argv[0]} csv_file upper_bound_txt_file gloro_number_txt_file output_pdf_file")
    sys.exit(1)

csv_file=sys.argv[1]
upper_bound_txt_file=sys.argv[2]
gloro_txt_file=sys.argv[3]
pdf_file=sys.argv[4]

upper_bound=0.0
with open(upper_bound_txt_file, 'r') as f:
    upper_bound=float(f.read().strip())

gloro_robustness=0.0
with open(gloro_txt_file, 'r') as f:
    gloro_robustness=float(f.read().strip())

file_path = csv_file
data = pd.read_csv(file_path)

x = data.iloc[:, 0]  # First column for x-values
y = data.iloc[:, 1]  # Second column for y-values
y2 = data.iloc[:, 2]  # Third column for the second y-axis
y2 = y2 / (60) # convert seconds to minutes

# convert proportions to percentages
upper_bound=upper_bound * 100
gloro_robustness=gloro_robustness * 100
y = y * 100

fig, ax1 = plt.subplots(figsize=(4.5, 3.5))
y1max = y.max()
ax1.plot(x, y, label=f"Verified Robustness (max {y1max}%)", color="blue")
ax1.set_ylim(0.0,100.0)
ax1.set_xlim(1,10)
ax1.set_xlabel("Gram Iterations")
ax1.tick_params(axis='y', labelcolor="blue")
ax1.set_ylabel("Robustness Percentage")


ax1.axhline(gloro_robustness, color="blue", linestyle="--", label=f"Unverified Robustness ({gloro_robustness:.2f}%)")
ax1.axhline(upper_bound, color="blue", linestyle=":", label=f"Measured Robustness ({upper_bound:.2f}%)")

ax2 = ax1.twinx()
ax2.plot(x, y2, label="Certifier Running Time (minutes)", color="green", linestyle='-.')
ax2.set_ylabel("Minutes to Compute Bounds", color="green")
ax2.tick_params(axis='y', labelcolor="green")
y2max = y2.max()
#ax2.set_yscale('log')
ax2.set_ylim(0,20)

#plt.title(graph_title)

# Combine both legends to the right of the second y-axis
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower center', bbox_to_anchor=(0.5, -0.6), ncol=2)

#ax2.legend(loc="upper right",bbox_to_anchor=(0, 1.15))
#ax1.legend(loc="upper left",bbox_to_anchor=(0, 1.15))

plt.tight_layout(rect=[0, 0, 1.6, 1])

#plt.grid()

plt.savefig(pdf_file, format="pdf", bbox_inches="tight")

# Show the plot
#plt.show()
