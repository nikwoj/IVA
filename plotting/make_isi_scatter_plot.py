import matplotlib.pyplot as plt
from numpy import log
from sys import argv

'''
NOTE: To use, call with python in following manner:

   python make_isi_scatter_plot.py name_to_save_as name_for_data xaxis yaxis

The file should be organized as follows:
   (xnumber, ynumber)\n(xnumber2, ynumber2)\n...
'''

def make_scatter_plot(name, subjects, xaxis, yaxis) :
    xdata, ydata = get_data(subjects) 
    
    plt.plot(xdata, ydata, 'bo')
    
    plt.semilogy()
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    
    plt.show()
    plt.savefig(name)

def get_data(subjects) :
    xdata = []
    ydata = []
    
    fil = open(subjects, "r+")
    lines = fil.readline().split(', ')
    
    while lines != [''] :
         xdata.append(int(  lines[0][1:]))
         ydata.append(float(lines[1][:-2]))
         lines = fil.readline().split(', ')
    
    return xdata, ydata

if __name__ == "__main__" :
    _,name,subjects,xaxis,yaxis = argv
    make_scatter_plot(name, subjects, xaxis, yaxis)
