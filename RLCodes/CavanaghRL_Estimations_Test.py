import numpy as np
import matplotlib.pyplot as plt

# import sys
# sys.path.insert(1, './RL-And-DD/FunctionalConnectivityCodes/GrangerCausality')
# from  ConMod import *

from RLMod import *

# # # Load the Actions and Rewards list # # #

all_stim_action_list, all_stim_reward_list, prob_class = Local_ActionRewardExtractor(stim = 0)
SubList = np.arange(len(all_stim_action_list))

# # # Define a PS_Task_Dataset() object and initialize it # # #

PS_DS = PS_Task_Dataset()
PS_DS.Load_NonStandard_Behavioral_Data(all_stim_action_list, all_stim_reward_list, prob_class)

# # # Load Loss and Gain Learning Rates # # #

a_Gs, a_Ls = Local_Load_Cavanagh_EstParams(SubList)

# # # Calculate the Log-Likelihood Values # # #

LLs = PS_DS.ParametersLikelihood(sublist = SubList, T = [0.2], a_gain = a_Gs, a_loss = a_Ls)

# # # Generate Random Data # # #

LL_n = np.zeros_like(SubList)
N = 10

for _ in range(N):

    a_Gn = np.random.uniform(0, 1, size = len(SubList))
    a_Ln = np.random.uniform(0, 1, size = len(SubList))

    LL_n = LL_n + 1 / N * np.array(PS_DS.ParametersLikelihood(sublist = SubList, T = [0.2], a_gain = a_Gn, a_loss = a_Ln))

# # # Plot Log-Likelihood Values Distribution # # #

fig, axs = plt.subplots(3, 1, layout = 'constrained', sharey = True, sharex = True, figsize = (7, 7))

axs[0].hist(LLs, label = 'Cavanagh Estimation', color = 'b')
axs[0].hist(LL_n, label = 'Random', color = 'r')
axs[0].legend()

axs[1].hist(LLs, label = 'Cavanagh Estimation', color = 'b')

axs[2].hist(LL_n, label = 'Random', color = 'r')

fig.suptitle('Histogram of Log-likelihood Values')

fig.show()
plt.show()

# # # Plot the Paper Estimated Values # # #

fig, axs = plt.subplots(3, 1, layout = 'constrained', sharey = True, sharex = True, figsize = (7, 7))

axs[0].hist(a_Gs, label = 'a_gain', color = 'b')
axs[0].hist(a_Ls, label = 'a_loss', color = 'r')
axs[0].legend()

axs[1].hist(a_Gs, label = 'a_gain', color = 'b')

axs[2].hist(a_Ls, label = 'a_loss', color = 'r')

fig.suptitle('Histogram of Cavanagh Paper\'s Learning Rate Estimations')

fig.show()
plt.show()