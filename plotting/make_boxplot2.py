from sys import argv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

def get_data(fil) :
    data = []
    fil = open(fil, "r+")
    text = fil.readline()
    while text != "" :
        data.append(float(text))
        text = fil.readline()
    
    return data

numBoxes = 16
xaxis = []
Names = ["PCA IVAG, ","PCA, ","RP IVAG, ","RP, "]
for i in [4,8,16,32] :
    for j in Names :
        xaxis.append(j + str(i))

data = []

argv.pop(0)
for fil in argv :
    data.append(get_data(fil))
    print fil


##
## matplotlib.org/examples/pylab_examples/boxplot_demo2.html
##
fig, ax1 = plt.subplots(figsize=(10, 6))
fig.canvas.set_window_title('A Boxplot Example')
plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

bp = plt.boxplot(data, notch=0, sym='+', vert=1, whis=1.5, widths=1)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

# Hide these grid behind plot objects
ax1.set_axisbelow(True)
ax1.set_title('Comparison of various methods ISI values')
ax1.set_xlabel('Preprocessing and Number of subjects')
ax1.set_ylabel('LOG(ISI)')
ax1.set_yscale('log')

## Now fill the boxes with desired colors
#boxColors = ['darkkhaki', 'royalblue']
#numBoxes = numDists*2
#medians = list(range(numBoxes))
#for i in range(numBoxes):
#    box = bp['boxes'][i]
#    boxX = []
#    boxY = []
#    for j in range(5):
#        boxX.append(box.get_xdata()[j])
#        boxY.append(box.get_ydata()[j])
#    boxCoords = list(zip(boxX, boxY))
#    # Alternate between Dark Khaki and Royal Blue
#    k = i % 2
#    boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
#    ax1.add_patch(boxPolygon)
#    # Now draw the median lines back over what we just filled in
#    med = bp['medians'][i]
#    medianX = []
#    medianY = []
#    for j in range(2):
#        medianX.append(med.get_xdata()[j])
#        medianY.append(med.get_ydata()[j])
#        plt.plot(medianX, medianY, 'k')
#        medians[i] = medianY[0]
#    # Finally, overplot the sample averages, with horizontal alignment
#    # in the center of each box
#    plt.plot([np.average(med.get_xdata())], [np.average(data[i])],
#             color='w', marker='*', markeredgecolor='k')

# Set the axes ranges and axes labels
ax1.set_xlim(0.5, numBoxes + 0.5)
top = 1.0
bottom = -1.0
ax1.set_ylim([0.01, 1])
xtickNames = plt.setp(ax1, xticklabels=xaxis)
#plt.setp(xtickNames, rotation=45, fontsize=8)
plt.setp(xtickNames, rotation=45, fontsize=8)
# Due to the Y-axis scale being different across samples, it can be
# hard to compare differences in medians across the samples. Add upper
# X-axis tick labels with the sample medians to aid in comparison
# (just use two decimal places of precision)
pos = np.arange(numBoxes) + 1
#upperLabels = [str(np.round(s, 2)) for s in medians]
#weights = ['bold', 'semibold']
#for tick, label in zip(range(numBoxes), ax1.get_xticklabels()):
#    k = tick % 2
#    ax1.text(pos[tick], top - (top*0.05), upperLabels[tick],
#             horizontalalignment='center', size='x-small', weight=weights[k],
#             color=boxColors[k])

# Finally, add a basic legend

plt.show()
fig.savefig('test.png')
