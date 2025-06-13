#PLOT Coeficient variation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend import Legend
import matplotlib.gridspec as gridspec


from PropSourceR3D_Target_CO import PropSourceR3D_Target_CO

fp = PropSourceR3D_Target_CO("Dataset_Conceiçao_4S_Py.xlsx")  #with Po
#fp = PropSourceR_Target_CO("sampledata.xlsx")
fp.infos()

nsample=[]
nsample = fp.nsample(nsample)
print('nsample', nsample)

nCB = nsample[0]  # max
nUR = nsample[1]  # max
nCF = nsample[2] # max
nNG = nsample[3] # max
nY = nsample[4] # max

import numpy as np
#Choose the sample subset for reduction
n=50 #times
m=0.1
#redus=[m,(2*m),round(3*m,2),(4*m),(5*m),round(6*m,2),round(7*m,2),(8*m),(9*m),10*m]
redus=[0.15,0.25,0.50,0.75,1.0]
#(m/2),,,(4*m),(5*m),round(6*m,2),round(7*m,2),(8*m),(9*m),10*m
print('redus=',redus)

CBcv2 = fp.multi_runs(n,nY, nCB, nUR, nCF,nNG,redus)

# coeficient variation for each target
n = 50
sns.set_style("whitegrid")

# read files
dados_CV_CO = pd.read_csv(f"Resume_CV2_Sources_Run{n}_CO.csv")

# dados_CV_CF = pd.read_csv(f"Resume_CV2_Run{n}_CF_5p.csv")


for i in range(1, 38):  # nun=1...38, where the target number is equal to 37
    dadosCO = dados_CV_CO[dados_CV_CO["Target"] == i]
    # X1_2 = np.array(dadosCB['propss'].values) #ascending = True
    # Y1_2 = np.array(dadosCB['CV_2'].values) #ascending = True

    COOrd = dadosCO.sort_values(by='nCB', ascending=False)  # Ordena a lista
    X1_2 = np.array(COOrd['propss'].values)
    Y1_2 = np.array(COOrd['CV_2'].values)

    # print(CBOrd)

    # create figure
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), sharey=True, sharex=False)
    # plt.xticks(np.arange(0, 20, 10))
    fig.suptitle("Target" + ' ' + str(i) + ' ')

    ax.plot(X1_2, Y1_2, "ko-", label="All sources", fillstyle='none', linewidth=0.8)
    # ax.plot(X2_2, Y2_2, "ks-",label="nUR",fillstyle='none',linewidth=0.8)
    # ax.plot(X3_2, Y3_2, "k^-",label="nCF",fillstyle='none',linewidth=0.8)
    plt.grid()
    plt.tight_layout()
    ax.set_xlabel("Percentage of samples")
    ax.set_ylabel('CV%, ' + str(n) + ' runs: 95% Confidence Region of CB,UR')
    # ax.legend(['nCB','nUR','nCF'], loc='upper right')
    plt.legend(loc='upper left')

    plt.savefig('Red_Coeff_Var_AllSources_runs=' + str(n) + '_Target=' + str(i) + '.png', dpi=300, bbox_inches='tight')
    plt.savefig('Red_Coeff_Var_AllSources_runs=' + str(n) + '_Target=' + str(i) + '.pdf', bbox_inches='tight')

# coeficient variation for All target
n = 50
sns.set_style("whitegrid")

# read files
dados_CV_CO = pd.read_csv(f"Resume_CV2_Sources_Run{n}_CO.csv")
tar = 37


# define the figure size and grid layout properties

def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


# create figure
fig, axs = plt.subplots(10, 4, figsize=(6, 10), sharex=True,
                        constrained_layout=True)  # 7 rows of 4 plots each to make 26 targets
axs = axs.ravel()  # Flatten the array of axes
axs = trim_axs(axs, tar)
plt.grid()
# for ax in enumerate(axs.flat):

for i in range(1, 38):  # nun=1...38, where the target number is equal to 37
    dadosCO = dados_CV_CO[dados_CV_CO["Target"] == i]
    # X1_2 = np.array(dadosCB['propss'].values) #ascending = True
    # Y1_2 = np.array(dadosCB['CV_2'].values) #ascending = True

    COOrd = dadosCO.sort_values(by='nCB', ascending=False)  # Ordena a lista
    X1_2 = np.array(COOrd['propss'].values)
    Y1_2 = np.array(COOrd['CV_2'].values)

    axs[i - 1].plot(X1_2, Y1_2, "ko-", fillstyle='none', linewidth=0.8)
    axs[i - 1].set_title("Target" + ' ' + str(i) + ' ', fontsize='small')
    # axs[i-1].set_ylabel('CV%, '+str(n)+' runs: 95% Confidence Region of CB,UR')
    # axs[i-1].set_xlabel('')
    # axs[i-1].set_axis_off()
    # if i==3:
    # axs[i].set_ylabel('CV%, '+str(n)+' runs: 95% Confidence Region of CB,UR')

    # axs[i-1].legend(bbox_to_anchor =[1.0, 1.1], prop = {"size": 9},loc='best', title = "% samples")

fig.supxlabel('Percentage of samples')
fig.supylabel('CV%, ' + str(n) + ' runs: 95% Confidence Region of CB,UR')
# for ax in axs.flat:
# ax.set_yticklabels([])

# fig.subplots_adjust(hspace=0.4)
# for ax in axs.flat:
# ax.label_outer()


plt.savefig('Red_Sources_Coeff_Var_F1_runs=' + str(n) + '_Target=' + str(i) + '.png', dpi=300, bbox_inches='tight')
plt.savefig('Red_Sources_Coeff_Var_F1_runs=' + str(n) + '_Target=' + str(i) + '.pdf', bbox_inches='tight')
