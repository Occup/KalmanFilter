#!/usr/bin/env python
import numpy as np
import KalmanFilter as KF

INIT_STATE = [27, 40]
STATE_TRANSFORMATION = [[1, 0], [0, 1]]
NOISE_TRANSFORMATION = [[0, 0], [0, 0]]
OBSERVATION = [[1, 0], [0, 1]]
Q = [[1, 0], [0, 1]]
R = [[0.5, 0], [0, 0.25]]
P = [[1, 0], [0, 1]]

SYS = [STATE_TRANSFORMATION, NOISE_TRANSFORMATION, OBSERVATION]
COV = [Q, R]
W0 = 0.025
test = KF.UnscentKalmanFilter(SYS, COV, P, INIT_STATE, OMEGA=(0.1, 2, 0))
trueval = []

for k in range(1000):
    test.predict()
    test.correct(np.random.normal([23.5, 36.5], [0.5, 0.25]))
    trueval.append(np.array([23.5, 36.5]))
test.show(trueval, ['tem', 'Tem'])
KF.plt.show()
