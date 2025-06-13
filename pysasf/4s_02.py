#PLOT Coeficient variation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend import Legend
import matplotlib.gridspec as gridspec

from matplotlib import ticker
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter

n = 50
sns.set_style("whitegrid")

# read files
dados_CV = pd.read_csv(f"Resume_CV2_Sources_Run{n}_CO.csv")
tar = [21, 22, 25, 26]

dadosT21 = dados_CV[dados_CV["Target"] == 21]

dadosT22 = dados_CV[dados_CV["Target"] == 22]

dadosT25 = dados_CV[dados_CV["Target"] == 25]

dadosT26 = dados_CV[dados_CV["Target"] == 26]

T25Ord = dadosT25.sort_values(by='nCB', ascending=False)  # Ordena a lista
X3_2 = np.array(T25Ord['propss'].values)
Y3_2 = np.array(T25Ord['CV_2'].values)
T26Ord = dadosT26.sort_values(by='nCB', ascending=False)  # Ordena a lista
X4_2 = np.array(T26Ord['propss'].values)
Y4_2 = np.array(T26Ord['CV_2'].values)
T21Ord = dadosT21.sort_values(by='nCB', ascending=False)  # Ordena a lista
X1_2 = np.array(T21Ord['propss'].values)
Y1_2 = np.array(T21Ord['CV_2'].values)
T22Ord = dadosT22.sort_values(by='nCB', ascending=False)  # Ordena a lista
X2_2 = np.array(T22Ord['propss'].values)
Y2_2 = np.array(T22Ord['CV_2'].values)

# create figure

fig, ax = plt.subplots(1, 1, figsize=(6, 4), sharey=True, sharex=False)

plt.ylim([-2, 80])
plt.yticks(np.arange(0, 85, 5))

ax.plot(X1_2, Y1_2, "ko-", label="Target 21", fillstyle='none', linewidth=0.8)
ax.plot(X2_2, Y2_2, "ks-", label="Target 22", fillstyle='none', linewidth=0.8)
ax.plot(X3_2, Y3_2, "k^-", label="Target 25", fillstyle='none', linewidth=0.8)
ax.plot(X4_2, Y4_2, "kx-", label="Target 26", fillstyle='none', linewidth=0.8)

plt.tight_layout()
ax.set_xlabel("Percentage of samples")
# ax.set_ylabel('CV%, '+str(n)+' runs: 95% Confidence Region on the CB\:UR-plane')
ax.set_ylabel('CV% of 95% Confidence Region on the (CB,UR) plane')
# ax.legend(['nCB','nUR','nCF'], loc='upper right')
plt.legend(loc='upper left')

plt.savefig('Red_Coef_Var_CO_run=' + str(n) + '_Sources=all' + '.png', dpi=300, bbox_inches='tight')
plt.savefig('Red_Coef_Var_CO_run=' + str(n) + '_Sources=all' + '.pdf', bbox_inches='tight')

plt.show()
# plt.close()


# coeficient variation for each target ####ARRUMAR
n = 50
sns.set_style("whitegrid")

# read files
dados_CV = pd.read_csv(f"Resume_CV2_Sources_Run{n}_GP.csv")
tar = [14, 15, 20, 21]

# dados_CV_CF = pd.read_csv(f"Resume_CV2_Run{n}_CF_5p.csv")


for i in tar:  # range(1, 27): #nun=1...27, where the target number is equal to 26
    data = dados_CV[dados_CV["Target"] == i]

    Coord = data.sort_values(by='nCB', ascending=False)  # Ordena a lista
    X1_2 = np.array(Coord['propss'].values)
    Y1_2 = np.array(Coord['CV_2'].values)

    # print(CBOrd)

    # create figure
    # plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), sharey=True, sharex=False)
    # plt.xticks(np.arange(0, 20, 10))
    # fig.suptitle("Target"+ ' '+str(i)+' ')

    ax.plot(X1_2, Y1_2, "ko-", label="All sources", fillstyle='none', linewidth=0.8)
    # ax.plot(X2_2, Y2_2, "ks-",label="nUR",fillstyle='none',linewidth=0.8)
    # ax.plot(X3_2, Y3_2, "k^-",label="nCF",fillstyle='none',linewidth=0.8)
    plt.grid()
    plt.tight_layout()
    ax.set_xlabel("Percentage of samples")
    ax.set_ylabel('CV%, ' + str(n) + ' runs: 95% Confidence Region of CB,UR')
    # ax.legend(['nCB','nUR','nCF'], loc='upper right')
    plt.legend(loc='upper left')

    plt.savefig('Red_Coeff_Var_GP_AllSources_runs=' + str(n) + '_Target=' + str(i) + '.png', dpi=300,
                bbox_inches='tight')
    plt.savefig('Red_Coeff_Var_GP_AllSources_runs=' + str(n) + '_Target=' + str(i) + '.pdf', bbox_inches='tight')

####ARRUMAR
# coeficient variation for All target
n = 50
sns.set_style("whitegrid")

# read files
dados_CV_CO = pd.read_csv(f"Resume_CV2_Sources_Run{n}_GP.csv")
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

for i in range(1, 27):  # nun=1...27, where the target number is equal to 26
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


plt.savefig('Red_Sources_Coeff_Var_GP_F1_runs=' + str(n) + '_Target=' + str(i) + '.png', dpi=300, bbox_inches='tight')
plt.savefig('Red_Sources_Coeff_Var_GP_F1_runs=' + str(n) + '_Target=' + str(i) + '.pdf', bbox_inches='tight')
