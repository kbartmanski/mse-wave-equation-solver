# import numpy
# import pylab
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle

# fig = plt.figure()
# ax = fig.add_subplot(111)

# im = [0 for i in range(6)]
# for i in range(6):
#     im[i], = ax.plot(range(6), pylab.randn(6))

# # create blank rectangle
# extra = Rectangle((0, 0), 0.5, 0.5, fc="w", fill=False, edgecolor='k', linewidth=0)

# #Create organized list containing all handles for table. Extra represent empty space
# legend_handle = [extra, extra, extra, extra, *im[0: 2], extra, *im[2: 4], extra, *im[4: 6]]

# #Define the labels
# label_col_1 = [r"$K \; \backslash \; p$", r"$2$", r"$\infty$"]
# label_j_1 = [r"$1$"]
# label_j_2 = [r"$2$"]
# label_j_3 = [r"$3$"]
# label_empty = [""]

# #organize labels for table construction
# legend_labels = numpy.concatenate([label_col_1, label_j_1, label_empty * 2, label_j_2, label_empty * 2, label_j_3, label_empty * 2])

# #Create legend
# ax.legend(legend_handle, legend_labels, 
#           loc = 9, ncol = 4, shadow = True, handletextpad = -1)

# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Create a sample plot
x = np.arange(1, 11)
y = x**2
plt.plot(x, y, marker='o')

# Create the table
# Define the cell text, here "" denotes the diagonal split cell
cell_text = [["", "1", "2", "3"],
             ["0.125", "", "", ""],
             ["0.25", "", "", ""]]

# Add a table at the bottom of the axes
the_table = plt.table(cellText=cell_text,
                      cellLoc='center',
                      loc='upper center',
                      bbox=[0.2, 0.5, 0.6, 0.4])

# Customizing table properties
the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
the_table.scale(1.2, 1.2)  # scaling the table

# Accessing the cell for diagonal line and setting background color
cell = the_table[0, 0]  # accessing the top-left cell
cell.get_text().set_text('c\\K')  # Adding text to the cell
cell.set_linewidth(2)  # Set the border width
# Adding diagonal line via path effect
from matplotlib.patheffects import withStroke
cell.get_text().set_path_effects([withStroke(linewidth=4, foreground="black")])

# Update cell backgrounds for clarity
# for i in range(3):
#     for j in range(1, 4):
#         the_table[i+1, j].set_facecolor("#f1f1f1")

# Display the plot
plt.show()