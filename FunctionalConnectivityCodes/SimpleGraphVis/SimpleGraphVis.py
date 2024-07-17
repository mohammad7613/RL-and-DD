import numpy as np
import matplotlib.pyplot as plt
import time

class DynamicGraph_NDW: # To Construct a Dynamic Undirected Weighted Graph
    
    def __init__(self, dyn_w_mat: np.ndarray):

        # Get Weight Matrix

        assert dyn_w_mat.shape[1] == dyn_w_mat.shape[2], 'Pass an NxN matrix to ...'
        
        self.weights = dyn_w_mat
            
    def get_coords(self, coords):

        # Get Coordinations to Visualize Network
        
        assert len(coords) == self.weights.shape[1], 'Coordinations must be available for each Node'

        self.coords = coords

    def get_times(self, times):

        # Get time Vector of Network Slices

        assert self.weights.shape[0] == len(times), 'time vector must has same length as number of temporal evolving matrices'

        self.t_ = times
        
    def draw_time_sample_graph(self, fig_size: list, win_number: int, VisThresh: int):

        # Draws a time sample,
        # win_number -> number of intended window
        # VisThresh -> Threshold to Consider Edge (it is useful to increase the sparsity of graph)
        
        plt.rcParams["figure.figsize"] = fig_size
        plt.rcParams["figure.autolayout"] = True

        x = self.coords[:, 0]
        y = self.coords[:, 1]

        for i in range(len(x)):

            plt.plot(x[i], y[i], marker="o", markeredgecolor="red", markerfacecolor="green")

            for j in range(i + 1, len(x)):

                color_num = self.weights[win_number, i, j] + self.weights[win_number, j, i]

                if color_num < 0:

                    color_num = 0

                elif color_num > 1:

                    color_num = 1

                if color_num > VisThresh:

                    plt.plot([x[i], x[j]], [y[i], y[j]], color = str(1 - color_num))
        
        plt.axis(False) 
        plt.show()

    def draw_dyn_graph(self, fig_size: list, delay: int, VisThresh: int):

        # Draws time evolution of a graph,
        # delay -> time delay between windows (because of refresh-rate probems, it doesn't work for small values)
        # VisThresh -> Threshold to Consider Edge (it is useful to increase the sparsity of graph

        plt.ion()

        fig = plt.figure(figsize = fig_size, )

        for i in range(self.weights.shape[0]):

            fig.canvas.draw()
            plt.title("Window time: " + str(np.floor(self.t_[i] * 100) / 100))
            self.draw_time_sample_graph(fig_size, i, VisThresh)
            fig.canvas.flush_events()
            time.sleep(delay)
            plt.clf()

class StaticGraph_NDNW: # To Construct a Static Undirected Non-Weighted Graph
    
    def __init__(self, laplacian_matrix: np.ndarray):

        assert laplacian_matrix.shape[0] == laplacian_matrix.shape[1], 'Pass an NxN matrix to ...'

        self.laplacian = laplacian_matrix
        
        self.weights = np.diag(np.diag(laplacian_matrix)) - laplacian_matrix
            
    def get_coords(self, coords):
        
        assert len(coords) == self.laplacian.shape[0], 'Coordinations must be available for each Node'

        self.coords = coords
        
    def draw_graph(self):
        
        plt.rcParams["figure.figsize"] = [20.00, 20.00]
        plt.rcParams["figure.autolayout"] = True

        x = self.coords[:, 0]
        y = self.coords[:, 1]

        for i in range(len(x)):

            plt.plot(y[i], x[i], marker="o", markeredgecolor="red", markerfacecolor="green")

            for j in range(i + 1, len(x)):

                if self.weights[i, j] != 0:

                    x_, y_ = line_generator(y[i], x[i], self.coords[j, 1], self.coords[j, 0], 1000)
        #             print(len(x_), len(y_))
                    plt.plot(x_[0 : np.min((len(x_), len(y_)))], y_[0 : np.min((len(x_), len(y_)))], color = 'black')
            
        plt.show()
            
    def plot_signal(self, signal, fontsize_):

        # It is useful for plotting a signal on nodes
        
        plt.rcParams["figure.figsize"] = [20.00, 20.00]
        plt.rcParams["figure.autolayout"] = True

        x = self.coords[:, 0]
        y = self.coords[:, 1]

        for i in range(len(x)):

            plt.plot(y[i], x[i], marker="o", markeredgecolor="red", markerfacecolor="green")
            plt.text(y[i], x[i], str(signal[i]), fontsize = fontsize_)

            for j in range(i + 1, len(x)):

                if self.laplacian[i, j] != 0:

                    x_, y_ = line_generator(y[i], x[i], self.coords[j, 1], self.coords[j, 0], 1000)
        #             print(len(x_), len(y_))
                    plt.plot(x_[0 : np.min((len(x_), len(y_)))], y_[0 : np.min((len(x_), len(y_)))], color = 'red')
        
        plt.show()
        
def line_generator(x1, y1, x2, y2, n):
    
    x_axis = np.arange(x1, x2, (x2 - x1) / n)
    y_axis = np.arange(y1, y2, (y2 - y1) / n)
    
    return x_axis, y_axis

class DynamicGraph_NDNW: # To Construct a Dynamic Undirected Non-Weighted Graph
    
    def __init__(self, dyn_w_mat: np.ndarray):

        assert dyn_w_mat.shape[1] == dyn_w_mat.shape[2], 'Pass an NxN matrix to ...'
        
        self.weights = dyn_w_mat
            
    def get_coords(self, coords):
        
        assert len(coords) == self.weights.shape[1], 'Coordinations must be available for each Node'

        self.coords = coords

    def get_times(self, times):

        assert self.weights.shape[0] == len(times), 'time vector must has same length as number of temporal evolving matrices'

        self.t_ = times
        
    def draw_time_sample_graph(self, fig_size: list, win_number: int):
        
        plt.rcParams["figure.figsize"] = fig_size
        plt.rcParams["figure.autolayout"] = True

        x = self.coords[:, 0]
        y = self.coords[:, 1]

        for i in range(len(x)):

            plt.plot(y[i], x[i], marker="o", markeredgecolor="red", markerfacecolor="green")

            for j in range(i + 1, len(x)):

                if self.weights[win_number, i, j] != 0:

                    plt.plot([x[i], x[j]], [y[i], y[j]])
        
        plt.axis(False) 
        plt.show()

    def draw_dyn_graph(self, fig_size: list, delay: int):

        plt.ion()

        
        fig = plt.figure(figsize = fig_size, )

        for i in range(self.weights.shape[0]):

            fig.canvas.draw()
            plt.title("Window time: " + str(np.floor(self.t_[i] * 100) / 100))
            self.draw_time_sample_graph(fig_size, i)
            fig.canvas.flush_events()
            time.sleep(delay)
            plt.clf()