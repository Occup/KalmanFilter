#!/usr/bin/env python
"""
this module is writen by Wang Qi(2019-05-29)
which implements high order linear-nonlinear Kalman Filter Algorithm,
consists of basic Kalman Filter,Exended Kalman Filter,Unscent Kalman Filter
and Square-Root Unscent Kalman Filter,
Basic Kalman Filter only supports Linear System,
All the other Variants of Kalman Filter support both linear and nonlinear System
"""

from choldate import cholupdate, choldowndate
import numpy as np
import sympy as sm
import matplotlib.pyplot as plt

# def cholupdate(R, x, sign='+'):
#     p = np.size(x)
#     x = x.T
#     for k in range(p):
#         if sign == '+':
#           r = np.sqrt(R[k, k]**2 + x[k]**2)
#         elif sign == '-':
#           r = np.sqrt(R[k, k]**2 - x[k]**2)
#         c = r/R[k, k]
#         s = x[k]/R[k, k]
#         R[k, k] = r
#         if sign == '+':
#           R[k, k+1:p] = (R[k, k+1:p] + s*x[k+1:p])/c
#         elif sign == '-':
#           R[k, k+1:p] = (R[k, k+1:p] - s*x[k+1:p])/c
#         x[k+1:p] = c*x[k+1:p] - s*R[k, k+1:p]
#     return R
  
class KalmanFilter():
    """
    Sparse Kalman Filter
    (system_model, covariance_matrix, P, Stat0)
    system_model = [status_transform_matrix, error_transform_matrix, observation_matrix]
    """
    @staticmethod
    def convert(data):
        """
        convert raw python data to numpy matrix type
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        else:
            pass
        return data

    def __init__(self, system_model, covariance_matrix, P, Stat0):
        self.phi = KalmanFilter.convert(system_model[0])
        self.gamma = KalmanFilter.convert(system_model[1])
        self.h_mat = KalmanFilter.convert(system_model[2])
        self.q_mat = KalmanFilter.convert(covariance_matrix[0])
        self.r_mat = KalmanFilter.convert(covariance_matrix[1])
        self.p_mat = KalmanFilter.convert(P)
        self.state = KalmanFilter.convert(Stat0)
        self.output = 0
        self.dataset = [self.state]
        self.measure = []

    def predict(self):
        """
        One step predict
        """
        self.state = self.phi @ self.state
        self.output = self.h_mat @ self.state

    def correct(self, new_measure):
        """
        use new measurement to calibrate state
        """
        z_state = KalmanFilter.convert(new_measure)
        p_mat = self.phi @ self.p_mat @ self.phi.T + \
                self.gamma @ self.q_mat @ self.gamma.T
        k_mat = p_mat @ self.h_mat.T @ np.linalg.inv(self.h_mat @ p_mat @ self.h_mat.T + self.r_mat)
        self.state = self.state + k_mat @ (z_state - self.output)
        self.p_mat = p_mat - k_mat @ self.h_mat @ p_mat
        self.dataset.append(self.state)
        self.measure.append(z_state)

    def show(self, trueval=None, label=None, savefig=False):
        """
        to plot all the data computed,
        Accept ideal result for plot,
        """
        ndim = len(self.measure[0])
        filtered = np.array([self.h_mat @ state for state in self.dataset])
        self.measure = KalmanFilter.convert(self.measure)
        if trueval is not None:
            trueval = np.array(trueval)
        for seq in range(ndim):
            f = plt.figure()
            plt.plot(self.measure[:, seq], 'k+', label='measurement')
            if trueval is not None:
                plt.plot(trueval[:, seq], 'g', label='True Value')
            plt.plot(filtered[:, seq], 'b--', label='filered output')
            plt.legend()
            plt.grid()
            plt.xlabel('Iteration')
            plt.ylabel('Value')
            plt.title(label[seq] if label else 'Dimension ' + str(seq+1))
            if savefig:
                f.savefig((f.axes[0].get_title() + '.png').replace(' ', '_'), dpi=600)
                

class SystemFunc():
    """
    A class used as nonlinear func and merge with ndarray computing
    give a array of symbols and expressions,which must be of the same length
    eg. symbols = [x,y,z]
        argument = [cos(x**2), sin(y**2), exp(z**2)]
    """
    def __init__(self, symbols_in_use, arguments):
        self.symbol = sm.Matrix(symbols_in_use)
        self.argument = sm.Matrix(arguments)
        self.jacobian = self.argument.jacobian(self.symbol)

    def jacob(self, given_state):
        """
        to convert a Jacobian Matrix to numpy Array
        """
        state = np.array(given_state)
        substitution = dict(zip(self.symbol, state))
        jacobian = np.array(self.jacobian.evalf(subs=substitution), dtype=float)
        return jacobian

    def propagate(self, given_state):
        """
        to propagate with nonlinear function
        """
        state = np.array(given_state)
        substitution = dict(zip(self.symbol, state))
        result = np.array(self.argument.evalf(subs=substitution), dtype=float).flatten()
        return result

class ExtendedKalmanFilter(KalmanFilter):
    """
    Extended Kalman Filter
    (system_model, covariance_matrix, P, Stat0)
    system_model = [status_transform_fucntion/matrix,
                    error_transform_function/matrix,
                    observation_function/matrix]
    """
    def __init__(self, system_model, covariance_matrix, P, Stat0):
        if isinstance(system_model[0], SystemFunc):
            self.phi_func = system_model[0]
            system_model[0] = system_model[0].jacob(Stat0)
        else:
            self.phi_func = None
        if isinstance(system_model[1], SystemFunc):
            self.gamma_func = system_model[1]
            system_model[1] = system_model[1].jacob(Stat0)
        else:
            self.gamma_func = None
        if isinstance(system_model[2], SystemFunc):
            self.h_mat_func = system_model[2]
            system_model[2] = system_model[2].jacob(Stat0)
        else:
            self.h_mat_func = None
        super().__init__(system_model, covariance_matrix, P, Stat0)

    def predict(self):
        """
        One step predict with nonlinear system model
        """
        if self.gamma_func is not None:
            self.gamma = self.gamma_func.jacob(self.state)
        if self.phi_func is not None:
            self.phi = self.phi_func.jacob(self.state)
            self.state = self.phi_func.propagate(self.state)
        else:
            self.state = self.phi @ self.state
        if self.h_mat_func is not None:
            self.output = self.h_mat_func.propagate(self.state)
            self.h_mat = self.h_mat_func.jacob(self.state)
        else:
            self.output = self.h_mat @ self.state
            
    def show(self, trueval=None, label=None, savefig=False):
        """
        to plot all the data computed,
        Accept ideal result for plot,
        """
        ndim = len(self.measure[0])
        if self.h_mat_func is not None:
            filtered = np.array([self.h_mat_func.propagate(state) for state in self.dataset])
        else:
            filtered = np.array([self.h_mat @ state for state in self.dataset])
        self.measure = KalmanFilter.convert(self.measure)
        if trueval is not None:
            trueval = np.array(trueval)
        for seq in range(ndim):
            f = plt.figure()
            plt.plot(self.measure[:, seq], 'k+', label='measurement')
            if trueval is not None:
                plt.plot(trueval[:, seq], 'g', label='True Value')
            plt.plot(filtered[:, seq], 'b--', label='filered output')
            plt.legend()
            plt.grid()
            plt.xlabel('Iteration')
            plt.ylabel('Value')
            plt.title(label[seq] if label else 'Dimension ' + str(seq+1))
            if savefig:
                f.savefig((f.axes[0].get_title() + '.png').replace(' ', '_'), dpi=600)

class UnscentKalmanFilter(ExtendedKalmanFilter):
    """
    Unscent Kalman Filter
    (system_model, covariance_matrix, P, Stat0)
    system_model = [status_transform_fucntion/matrix,
                    error_transform_function/matrix,
                    observation_function/matrix]
    """
    @staticmethod
    def sigma_points(state, omega, p_mat):
        """
        generate sigma points around state with parameter omega and P Matrix
        state imply the vector to be add sigma points
        omega is a set of parameter [alpha, beta, lambda] which control the selection of sigma points
        p_mat is covariance matrix of current state
        """
        # print("P_matrix = \n", p_mat, "\n")
        alpha, beta, kappa = omega
        lamda = alpha**2*(len(state)+kappa)-len(state)
        # part, diag, _ = np.linalg.svd(p_mat)
        # a_mat = part @ np.diag(diag)
        a_mat = np.linalg.cholesky(p_mat)
        coenf = np.sqrt(len(state) + lamda)
        sigma_p = np.vstack((state, state + a_mat.T * coenf, state - a_mat.T * coenf))
        weight_c = np.hstack((lamda/(len(state)+lamda)+(1-alpha**2+beta),\
                              (1/2/(len(state)+lamda)) * np.ones(2*len(state))))
        weight_m = np.hstack((lamda/(len(state)+lamda),\
                              (1/2/(len(state)+lamda)) * np.ones(2*len(state))))
        return list(zip(weight_m, weight_c, sigma_p))

    def __init__(self, system_model, covariance_matrix, P, Stat0, OMEGA=(0.001, 2, 0)):
        self.omega = OMEGA
        super().__init__(system_model, covariance_matrix, P, Stat0)

    def predict(self):
        """
        One Step predict
        """
        if self.gamma_func is not None:
            self.gamma = self.gamma_func.jacob(self.state)
        sigma_x = UnscentKalmanFilter.sigma_points(self.state, self.omega, self.p_mat)
        if self.phi_func is not None:
            step_sigma_x = [[weight_m, weight_c, self.phi_func.propagate(point)]\
                            for weight_m, weight_c, point in sigma_x]
        else:
            step_sigma_x = [[weight_m, weight_c, self.phi @ point]\
                            for weight_m, weight_c, point in sigma_x]
        self.state = np.sum([weight * point\
                             for weight, _, point in step_sigma_x], axis=0)
        self.p_mat = np.sum([weight * np.outer((point - self.state), (point - self.state))\
                     for _, weight, point in step_sigma_x], axis=0) \
                     + self.gamma @ self.q_mat @ self.gamma.T

    def correct(self, new_measure):
        """
        use new measurement to calibrate state
        """
        z_state = KalmanFilter.convert(new_measure)
        sigma_x = UnscentKalmanFilter.sigma_points(self.state, self.omega, self.p_mat)
        if self.h_mat_func is not None:
            step_sigma_x = [[weight_m, weight_c, point, self.h_mat_func.propagate(point)]\
                            for weight_m, weight_c, point in sigma_x]
        else:
            step_sigma_x = [[weight_m, weight_c, point, self.h_mat @ point]\
                            for weight_m, weight_c, point in sigma_x]
        self.output = np.sum([weight * point\
                              for weight, _, _, point in step_sigma_x], axis=0)
        covarz = np.sum([weight * np.outer((point - self.state), (step_point - self.output))\
                         for _, weight, point, step_point in step_sigma_x], axis=0)
        s_mat = np.sum([weight * np.outer((step_point-self.output), (step_point-self.output))\
                        for _, weight, _, step_point in step_sigma_x], axis=0) + self.r_mat
        k_mat = covarz @ np.linalg.inv(s_mat)
        self.state = self.state + k_mat @ (z_state - self.output)
        self.p_mat = self.p_mat - k_mat @ s_mat @ k_mat.T
        self.dataset.append(self.state)
        self.measure.append(z_state)

class SquareRootUnscentKalmanFilter(ExtendedKalmanFilter):
    """
    Square-Root Unscent Kalman Filter
    Improved Algorithm to avoid a situation when P_matrix is not positive definite
    (system_model, covariance_matrix, P, Stat0)
    system_model = [status_transform_fucntion/matrix,
                    error_transform_function/matrix,
                    observation_function/matrix]
    """
    @staticmethod
    def sigma_points(state, omega, s_mat):
        """
        generate sigma points around state with parameter omega and S Matrix,
        where S=cholesky(P)
        state imply the vector to be add sigma points
        omega is a set of parameter [alpha, beta, lambda]
        which control the selection of sigma points
        p_mat is covariance matrix of current state
        """
        # print("P_matrix = \n", p_mat, "\n")
        alpha, beta, kappa = omega
        lamda = alpha**2 * (len(state)+kappa) - len(state)
        coenf = alpha * np.sqrt(len(state) + kappa)
        sigma_p = np.vstack((state, state + s_mat.T * coenf, state - s_mat.T * coenf))
        weight_c = np.hstack((lamda/(len(state)+lamda)+(1-alpha**2+beta),\
                              (1/2/(len(state)+lamda)) * np.ones(2*len(state))))
        weight_m = np.hstack((lamda/(len(state)+lamda),\
                              (1/2/(len(state)+lamda)) * np.ones(2*len(state))))
        return list(zip(weight_m, weight_c, sigma_p))

    def __init__(self, system_model, covariance_matrix, P, Stat0, OMEGA=(0.001, 2, 0)):
        self.omega = OMEGA
        super().__init__(system_model, covariance_matrix, P, Stat0)
        self.s_mat = np.linalg.cholesky(self.p_mat).transpose()
        self.r_mat = np.linalg.cholesky(self.r_mat)
        self.q_mat = np.linalg.cholesky(self.q_mat)

    def predict(self):
        """
        One Step predict
        """
        if self.gamma_func is not None:
            self.gamma = self.gamma_func.jacob(self.state)
        sigma_x = SquareRootUnscentKalmanFilter.sigma_points(self.state, self.omega, self.s_mat)
        if self.phi_func is not None:
            step_sigma_x = [[weight_m, weight_c, self.phi_func.propagate(point)]\
                            for weight_m, weight_c, point in sigma_x]
        else:
            step_sigma_x = [[weight_m, weight_c, self.phi @ point]\
                            for weight_m, weight_c, point in sigma_x]
        self.state = np.sum([weight * point\
                            for weight, _, point in step_sigma_x], axis=0)
        _, s_mat = np.linalg.qr(np.vstack((np.vstack([np.sqrt(weight)*(point-self.state)\
                                for _, weight, point in step_sigma_x[1:]]), self.q_mat)))
        cholupdate(s_mat, step_sigma_x[0][1]*(step_sigma_x[0][2]-self.state))
        self.s_mat = s_mat

    def correct(self, new_measure):
        """
        use new measurement to calibrate state
        """
        z_state = KalmanFilter.convert(new_measure)
        sigma_x = SquareRootUnscentKalmanFilter.sigma_points(self.state, self.omega, self.s_mat)
        if self.h_mat_func is not None:
            step_sigma_x = [[weight_m, weight_c, point, self.h_mat_func.propagate(point)]\
                            for weight_m, weight_c, point in sigma_x]
        else:
            step_sigma_x = [[weight_m, weight_c, point, self.h_mat @ point]\
                            for weight_m, weight_c, point in sigma_x]
        self.output = np.sum([weight * point\
                                  for weight, _, _, point in step_sigma_x], axis=0)
        _, s_mat = np.linalg.qr(np.vstack((np.vstack([np.sqrt(weight)*(step_point-self.output)\
                                for _, weight, _, step_point in step_sigma_x[1:]]), self.r_mat)))
        if step_sigma_x[0][1] >= 0:
            cholupdate(s_mat, np.sqrt(step_sigma_x[0][1])*(step_sigma_x[0][3]-self.output))
        else:
            choldowndate(s_mat, np.sqrt(-step_sigma_x[0][1])*(step_sigma_x[0][3]-self.output))
        covarz = np.sum([weight * np.outer((point - self.state), (step_point - self.output))\
                         for _, weight, point, step_point in step_sigma_x], axis=0)
        k_mat = covarz @ np.linalg.inv(s_mat) @ np.linalg.inv(s_mat.T)
        u_mat = k_mat @ s_mat.T
        self.state = self.state + k_mat @ (z_state - self.output)
        for col in u_mat.T:
            choldowndate(self.s_mat, col)
        self.dataset.append(self.state)
        self.measure.append(z_state)
