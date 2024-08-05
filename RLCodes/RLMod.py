import numpy as np
import matplotlib.pyplot as plt
import pickle
from pandas import read_csv, read_excel
import ast

class PS_Task_Pop_Simulator():

    ## may add imbalance feedbacks!
    ## Uploading an existing dataset!

    def __init__(self, population_number, prob_class, exp_length, punishment = False, ProbabilityCalculationMethod = 'softmax'):

        assert population_number > 4, "Population must be at least include 5 members"
        assert prob_class >= 0.5 and prob_class <= 1, "The prob_class must be between 0.5 and 1 (the reward probability of the more rewarded one)"

        self.population = population_number
        self.PrCl = prob_class
        self.EL = exp_length

        self.PunMiss = punishment
        self.ProbGenMet = ProbabilityCalculationMethod

    def AssignLearningRate(self, a):

        assert type(a) == float or type(a) == list or type(a) == np.ndarray, "Learning rate must be a float number or array of float numbers"

        if type(a) == float:

            print("Whole population have a similar learning rate")

            self.LR = a
            self.LR_HG = 0

        else:

            a = np.array(a)

            assert a.ndim < 3, "Learning rate array should have less than 3 dimensions"

            if a.ndim == 0:

                print("Whole population have a similar learning rate")

                self.LR = a[0]
                self.LR_HG = 0

            elif a.ndim == 1:

                assert a.shape[0] == self.population or a.shape[0] == 2, "Number of assigned learning rates doesn't match the population"

                if a.shape[0] == self.population:

                    print("You assign a different Learning Rate for each member")

                    self.LR = a
                    self.LR_HG = 1

                else:

                    if a.shape[0] == 2:

                        print("You assign different gain and loss Learning Rate")

                        self.LR = a
                        self.LR_HG = 2              

            elif a.ndim == 2:

                assert a.shape[0] == 2 or a.shape[1] == 2, "The assigned learning rates of each subject must be maiximum two"
                assert a.shape[0] == self.population or a.shape[1] == self.population, "Number of assigned learning rates doesn't match the population"

                if a.shape[0] == 2:

                    a = a.T

                self.LR = a
                self.LR_HG = 3

    def AssignTemperatureFactor(self, T):

        assert type(T) == float or type(T) == list or type(T) == np.ndarray, "TF must be a float number or array of float numbers"

        if type(T) == float:

            print("Whole population have a similar TF")

            self.TF = T
            self.TF_HG = 0

        else:

            T = np.array(T)

            assert T.ndim < 2, "Learning rate array should have less than 2 dimensions"

            if T.ndim == 0:

                print("Whole population have a similar TF")

                self.TF = T[0]
                self.TF_HG = 0

            elif T.ndim == 1:

                assert T.shape[0] == self.population, "Number of assigned learning rates doesn't match the population"

                print("You assign a different Learning Rate for each member")

                self.TF = T
                self.TF_HG = 1

    def Experiment_Generator(self): # Generate a random Experiment, as environment

    # # The 0 trials reward on the more rewarded stimulus as vice versa.

        return np.random.permutation(np.concatenate((np.zeros(int(self.PrCl * self.EL)), np.ones(int((1 - self.PrCl) * self.EL + 1)))))

    def Subject_Simul(self):

        # This Functions simulates a subject in the Probabilistic Learning Task with given Arguments
        # # Prob_Class -> Reward chance for the more rewarded stimulus
        # # Leangth -> The number of trials
        # # T -> Temperature factor, indicates the Explore/Exploit Tendency
        # # a -> The learning rate:
        # # # a: float number or single-index array/list -> a unique learning rate for both gain and loss
        # # # a: double-indexed array/list -> a = [a_gain, a_loss]. learning rates for gain and loss, respectively

        self.G_Choices = []
        self.G_Rewards = []
        self.G_Q = []

        G_Choices = []
        G_Rewards = []
        G_Q = []

        for mem in range(self.population):

            if self.LR_HG == 0:

                a = self.LR

            elif self.LR_HG == 1:

                a = self.LR[mem]

            elif self.LR_HG == 2:

                a = self.LR

            elif self.LR_HG == 3:

                a = self.LR[mem, :]

            else:

                print("This is a bug in the class code")

                return False

            if self.TF_HG == 0:

                T = self.TF

            elif self.TF_HG == 1:

                T = self.TF[mem]

            else:

                print("This is a bug in the class code")

                return False

            Experiment = self.Experiment_Generator()

            Q = np.array([0.0, 0.0])

            Choices = []
            Rewards = []

            for i, Trial in enumerate(Experiment):

                Choice = self.Decision_Maker(self.Choice_Probability(Q, 0, T), 0)

                Reward = self.Reward_Generator(Trial, Choice)

                if type(a) == float:

                    Q[Choice] = Q[Choice] + a * (Reward - Q[Choice])

                elif len(a) == 1:

                    Q[Choice] = Q[Choice] + a[0] * (Reward - Q[Choice])

                else:

                    Q[Choice] = Q[Choice] + a[0] * np.max([Reward - Q[Choice], 0]) + a[1] * np.min([Reward - Q[Choice], 0])

                Choices.append(Choice)
                Rewards.append(Reward)

            G_Choices.append(Choices)
            G_Rewards.append(Rewards)
            G_Q.append(Q)

        self.G_Choices = np.array(G_Choices)
        self.G_Rewards = np.array(G_Rewards)
        self.G_Q = np.array(G_Q)

    def Reward_Generator(self, Trial, Choice): # Generates the Reward of Choice in the given Trial (0 Choice reward on 0 and 1 on 1)

        if self.PunMiss:

            return 1 - np.abs(Trial - Choice)

        else:

            return 1 - 2 * np.abs(Trial - Choice)

    def Choice_Probability(self, Q, Choice, T): # Generate The Probability of Selecting a Stimulation

        if self.ProbGenMet == 'softmax':

            return np.exp(Q[Choice] / T) / (np.exp(Q[Choice] / T) + np.exp(Q[1 - Choice] / T))

        else:

            print("Only Softmax is defined ^-^")

            return False

    def Decision_Maker(self, Q_Prob, Choice):

        if np.random.random() < Q_Prob:

            return Choice
        
        else:

            return 1 - Choice

    def Accuracy_Performance(self, sublist, ax = None):

        assert type(sublist) == int or type(sublist) == list or type(sublist) == np.ndarray, "Subjects list/number must be integer or list/array of integers"

        if type(sublist) == int:

            assert sublist >= 0 and sublist < self.population, "Insert valid subject number"

            Acc = []

            for i in range(self.EL):

                Acc.append(1 - np.sum(self.G_Choices[sublist, :i]) / (i + 1))

            if ax != None:

                ax.plot(np.arange(self.EL), Acc)
                ax.set_xlabel("Trial Nmuber")
                ax.set_ylabel("Accuracy")
                ax.set_title("Accuracy Performance")

            return np.array(Acc)

        else:

            Acc = []

            sublist = np.array(sublist)

            assert sublist.ndim == 1, "List of Subjects must be 1-Dimensional"
            assert np.all(sublist < self.population) and np.all(sublist >= 0), "Invalid subject in list"

            for sub in sublist:

                Acc_sub = []

                for i in range(self.EL):

                    Acc_sub.append(1 - np.sum(self.G_Choices[sub, :i]) / (i + 1))

                if ax != None:

                    ax.plot(np.arange(self.EL), Acc_sub)
                    ax.set_xlabel("Trial Nmuber")
                    ax.set_ylabel("Accuracy")
                    ax.set_title("Accuracy Performance")

                Acc.append(Acc_sub)

            if ax != None:

                ax.plot(np.arange(self.EL), np.mean(np.array(Acc), axis = 0), label = 'Mean Performance', color = 'k')
                ax.legend()

            return np.array(Acc)

    def BuildAndSave(self, output_name, output_dir = r'E:\HWs\Msc\Research\Research\Reports\Presentations\Aug6\Behavioral\\'):

        self.Pop_Perf = {'PopNum': self.population, 'Exp Length': self.EL, 'Prob Class': self.PrCl, 'LR': self.LR, 'TempFact': self.TF, 'Actions': self.G_Choices, 'Feedbacks': self.G_Rewards}

        with open(output_dir + output_name, 'wb') as f:
            
            pickle.dump(self.Pop_Perf, f)

class PS_Task_Dataset():

    def __init__(self, StandardForm_Data_dir = None):

        if StandardForm_Data_dir != None:

            with open(StandardForm_Data_dir, 'rb') as f:
            
                self.Pop_Perf = pickle.load(f)

                # self.Pop_Perf = {'PopNum': self.population, 'Exp Length': self.EL, 'Prob Class': self.PrCl, 'LR': self.LR, 'TempFact': self.TF, 'Actions': self.G_Choices, 'Feedbacks': self.G_Rewards}

                self.population = self.Pop_Perf['PopNum']
                self.EL = self.Pop_Perf['Exp Length']
                self.PrCl = self.Pop_Perf['Prob Class']
                self.LR = self.Pop_Perf['LR']
                self.TF = self.Pop_Perf['TempFact']
                self.G_Choices = self.Pop_Perf['Actions']
                self.G_Rewards = self.Pop_Perf['Feedbacks']

                self.SFP = True

        else:

            self.SFP = False


    def Load_NonStandard_Behavioral_Data(self, action_list, reward_list, prob_class):

        assert self.SFP == False, "You have loaded the standard form data"
        assert len(action_list) == len(reward_list), "Inputs doesn't match"

        sub_trial_number = []

        for i in range(len(action_list)):

            actions = action_list[i]
            rewards = reward_list[i]

            assert len(actions) == len(rewards), "Subject " + str(i) + " has unmatch action and reward arrays"

            sub_trial_number.append(len(actions))

        self.G_Choices = action_list
        self.G_Rewards = reward_list
        self.EL = sub_trial_number
        self.population = len(action_list)
        self.PrCl = prob_class

    def Accuracy_Performance(self, sublist, ax = None):

        assert type(sublist) == int or type(sublist) == list or type(sublist) == np.ndarray, "Subjects list/number must be integer or list/array of integers"

        if type(sublist) == int:

            assert sublist >= 0 and sublist < self.population, "Insert valid subject number"

            Acc = []

            for i in range(self.EL[sublist]):

                Acc.append(1 - np.sum(self.G_Choices[sublist][:i]) / (i + 1))

            if ax != None:

                ax.plot(Acc)
                ax.set_xlabel("Trial Number")
                ax.set_ylabel("Accuracy")
                ax.set_title("Subject " + str(sublist) + " perfromance during trials")

            return Acc

        else:

            Acc = []

            sublist = np.array(sublist)

            assert sublist.ndim == 1, "List of Subjects must be 1-Dimensional"
            assert np.all(sublist < self.population) and np.all(sublist >= 0), "Invalid subject in list"

            if ax != None:

                print("We plot what you want, but trial lengths may be different!")
                ax.set_xlabel("Trial Number")
                ax.set_ylabel("Accuracy")

            for sub in sublist:

                Acc_sub = []

                for i in range(self.EL[sub]):

                    Acc_sub.append(1 - np.sum(self.G_Choices[sub][:i]) / (i + 1))

                if ax != None:

                    ax.plot(Acc_sub)

                Acc.append(Acc_sub)

            return Acc

    def ParametersLikelihood(self, sublist, **kwargs):

        options = {

            'T': [0.2],
            'a_gain': [0.1],
            'a_loss': [0.1]
        }

        assert len(kwargs) > 0, "Insert at least one and at most three parameter"
        assert type(sublist) == list or type(sublist) == np.ndarray, "Enter subjects as a list"

        if type(sublist) == list:

            sublist = np.array(sublist)

        assert np.all(sublist >= 0) and np.all(sublist < self.population), "The subjects list includes invalid IDs"

        options.update(kwargs)

        assert (type(options['T']) == list or type(options['T']) == np.ndarray) and (len(options['T']) == 1 or len(options['T']) == len(sublist)), "Insert a list/array of float number(s) for Temperature Factor, with length 1 or equal to population"
        assert (type(options['a_gain']) == list or type(options['a_gain']) == np.ndarray) and (len(options['a_gain']) == 1 or len(options['a_gain']) == len(sublist)), "Insert a list/array of float number(s) for Learning Rate, with length 1 or equal to population"
        assert (type(options['a_loss']) == list or type(options['a_loss']) == np.ndarray) and (len(options['a_loss']) == 1 or len(options['a_loss']) == len(sublist)), "Insert a list/array of float number(s) for Learning Rate, with length 1 or equal to population"

        testing_parameters = [options['T'], options['a_gain'], options['a_loss']]

        assert len(testing_parameters[1]) == len(testing_parameters[2]), "No TAXATION without REPRESENTATION"

        HG = [len(testing_parameters[i]) for i in range(2)]

        LL_values = []

        if HG[0]:

            if HG[1]:

                # Same parameters for all subjects

                for i, sub in enumerate(sublist):

                    LL_values.append(Likelihood_C(self.G_Choices[sub], self.G_Rewards[sub], T_est = testing_parameters[0][0], ag_est = testing_parameters[1][0], al_est = testing_parameters[2][0]))

            else:

                # Same Temperature but different Learning Rates

                for i, sub in enumerate(sublist):

                    LL_values.append(Likelihood_C(self.G_Choices[sub], self.G_Rewards[sub], T_est = testing_parameters[0][0], ag_est = testing_parameters[1][i], al_est = testing_parameters[2][i]))


        else:

            if HG[1]:

                # Same Learning Rate and different Temperature

                for i, sub in enumerate(sublist):

                    LL_values.append(Likelihood_C(self.G_Choices[sub], self.G_Rewards[sub], T_est = testing_parameters[0][i], ag_est = testing_parameters[1][0], al_est = testing_parameters[2][0]))


            else:

                # Different Parameters for all subjects

                for i, sub in enumerate(sublist):

                    LL_values.append(Likelihood_C(self.G_Choices[sub], self.G_Rewards[sub], T_est = testing_parameters[0][i], ag_est = testing_parameters[1][i], al_est = testing_parameters[2][i]))

        return LL_values

def Likelihood_C(Choices, Rewards, ag_est, al_est, T_est):

    Ps = []
    
    Q = np.array([0.0, 0.0])

    for i in range(len(Choices)):

        Choice = int(Choices[i])
        Reward = Rewards[i]

        P = Choice_Probability(Q, Choice, T_est)

        Q[Choice] = Q[Choice] + ag_est * np.max([Reward - Q[Choice], 0]) + al_est * np.min([Reward - Q[Choice], 0])

        Ps.append(P)

    return np.sum(np.log(Ps))

def Choice_Probability(Q, Choice, T):

    return np.exp(Q[Choice] / T) / (np.exp(Q[Choice] / T) + np.exp(Q[1 - Choice] / T))

######################################################## Locally Adjusted Functions :) #################################################

def Local_ActionRewardExtractor(stim: int, dataframe_dir = r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\Subjects_Behavioral_datas.csv'):
    
    dataframe = read_csv(dataframe_dir)

    all_stim_action_list = []
    all_stim_reward_list = []

    for sample in range(len(dataframe)):
        
        list_ = ast.literal_eval(dataframe['Task'][sample])
        Tasks_matrix = np.array(list_)

        all_stim_action_list.append(np.squeeze(Tasks_matrix[np.where(Tasks_matrix[:, 0] == stim), 2]))
        all_stim_reward_list.append((np.squeeze(Tasks_matrix[np.where(Tasks_matrix[:, 0] == stim), 3]) + 1) / 2)

    prob_class = 0.8 - stim * 0.1

    return all_stim_action_list, all_stim_reward_list, prob_class

def Local_Load_Cavanagh_EstParams(AvailableSubjects, excel_dir = r'E:\HWs\Msc\Research\Research\Depression Dataset\depression_rl_eeg\Depression PS Task\Scripts from Manuscript\Data_4_Import.xlsx'):

    tmp_Data = read_excel(excel_dir)

    return np.array(tmp_Data['TST_aG'][AvailableSubjects]), np.array(tmp_Data['TST_aL'][AvailableSubjects])