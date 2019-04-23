#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def convert(data):
    """
    convert raw python data to numpy matrix type
    """
    if isinstance(data, np.matrix):
        if data.shape[0] == 1 and data.shape[1] != 1:
            data_converted = data.T
    else:
        data_converted = np.matrix(data)
        if data_converted.shape[0] == 1 and data_converted.shape[1] != 1:
            data_converted = data_converted.T
    return data_converted
    
class KalmanFilter():
    """
    Sparse Kalman Filter
    (system_model, covariance_matrix, P, Stat0)
    system_model = [status_transform_matrix, error_transform_matrix, observation_matrix]
    """
    def __init__(self, system_model, covariance_matrix, P, Stat0):
        self.phi = convert(system_model[0])
        self.gamma = convert(system_model[1])
        self.h_mat = convert(system_model[2])
        self.q_mat = convert(covariance_matrix[0])
        self.r_mat = convert(covariance_matrix[1])
        self.p_mat = convert(P)
        self.state = convert(Stat0)
        self.dataset = [self.state]
        self.measure = []


    def predict(self):
        """
        One step predict
        """
        self.state = self.phi * self.state

    def correct(self, new_measure):
        """
        use new measurement to calibrate state
        """
        z_state = convert(new_measure)
        p_mat = self.phi * self.p_mat * self.phi.T + \
                self.gamma * self.q_mat * self.gamma.T
        k_mat = p_mat * self.h_mat.T * np.linalg.inv(self.h_mat * p_mat * self.h_mat.T + self.r_mat)
        self.state = self.state + k_mat * (z_state - self.h_mat * self.state)
        self.p_mat = p_mat - k_mat * self.h_mat * p_mat
        self.dataset.append(self.state)
        self.measure.append(z_state)

    def show(self, trueval=None):
        """
        to plot all the data computed,
        Accept ideal result for plot,
        """
        for seq in range(self.measure[0].shape[0]):
            plt.figure(seq+1)
            mes = [self.measure[k][(seq, 0)] for k in range(len(self.measure))]
            dat = [self.dataset[k][(seq, 0)] for k in range(len(self.dataset))]
            if trueval is not None:
                tru = [trueval[k][(seq, 0)] for k in range(len(trueval))]\
                    if trueval[0].shape[1] == 1 else \
                       [trueval[k][(0, seq)] for k in range(len(trueval))]
            plt.plot(mes, 'k+', label='measurement')
            plt.plot(dat, 'b-', label='posteriestimate')
            if trueval is not None:
                plt.plot(tru, 'g', label='True Value')
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('Value')
            plt.title('Dimension '+str(seq+1))
        plt.show()
