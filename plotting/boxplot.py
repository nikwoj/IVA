import glob
import seaborn as sb
from matplotlib import pyplot as plt
import pandas as pd

subjects = [4, 8, 16, 32]
methods  = {'IVAG_pooled_PCA': 'PCA IVAG IVAL',
            'noIVAG_pooled_PCA': 'PCA IVAL',
            'IVAG_pooled_rand_proj': 'RP IVAG IVAL',
            'noIVAG_pooled_rand_proj': 'RP IVAL'}

df = pd.DataFrame()

for m in methods:
    dm = pd.DataFrame()
    for s in subjects:
        d = pd.read_table(glob.iglob('*_'+m+'*'+'%03d'%s+'*').next(),
                              header=None, names=[methods[m]])
        d['subj'] = s
        dm = dm.append(d, ignore_index=True)
    print m
    df[methods[m]] = dm[methods[m]]
df['subj']=dm['subj']

d = pd.DataFrame()
for m in methods:
    t = pd.DataFrame()
    t['errors'] = df[methods[m]]
    t['subj'] = df['subj']
    t['method'] = methods[m]
    d = d.append(t, ignore_index=True)

shift = 0.15
wds = 0.8
fliersz = 2
lwd = 1

plt.figure()
ax = sb.boxplot(x="subj", y="errors", hue="method",
                    hue_order=['PCA IVAG IVAL','PCA IVAL','RP IVAG IVAL','RP IVAL'],
                    data=d,
                    palette="Set1",
                    linewidth=lwd,
                    width=wds,
                    fliersize=fliersz)
plt.show()
