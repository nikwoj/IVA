from sys import argv
from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes

def get_data(fil) :
    fil = open(fil, "r+")
    data = []
    text = fil.readline()
    while text != "" :
        data.append(float(text))
        text = fil.readline()
    
    return data

fig = figure()
ax = axes()
hold(True)

colors = ['#33a02c', '#b2df8a', '#1f78b4', '#a6cee3']

argv.pop(0)
i = 0
j = 0
data = []
for fil in argv :
    i += 1
    data.append(get_data(fil))
    print fil
    if i%4 == 0 :
        bp = boxplot(data, positions=[i+j-3, i+j-2, i+j-1, i+j], widths=1, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors) :
            patch.set(color='black', linewidth=0.5)
            patch.set_facecolor(color)
        
        for median in bp['medians'] :
            median.set(color='black')
        for whisker in bp['whiskers'] :
            whisker.set(color='black')
        j += 1
        data = []
    

#for patch, color in zip(bp['boxes'], colors) :
#    patch.set_facecolor(color)

# set axes limits and labels
ax.set_yscale('log')
ax.set_title("Comparison of various method's ISI values")
ax.set_xlabel("Number of Subjects")
ax.set_ylim([0.015,1])
ax.set_xlim([0,20])
ax.set_xticklabels(['4','8','16','32'])
ax.set_xticks([2.5, 7.5, 12.5, 17.5])



hc, = plot([1], '#33a02c')
hl, = plot([1,1], '#b2df8a')
ht, = plot([1,1], '#1f78b4')
hp, = plot([1,1], '#a6cee3')
legend((hc,hl,ht,hp),("PCA IVAG", "PCA", "RP IVAG", "RP"), loc='best')
hc.set_visible(False)
hl.set_visible(False)
ht.set_visible(False)
hp.set_visible(False)

# draw temporary red and blue lines and use them to create a legend
#hB, = plot([1,1],'b-')
#hR, = plot([1,1],'r-')
#legend((hB, hR),('Apples', 'Oranges'))
#hB.set_visible(False)
#hR.set_visible(False)

savefig('boxcompare.png')
show()
