#!/usr/bin/env python
import Kalman as KF
import numpy as np

INIT_STATE = [27, 40]
STATE_TRANSFORMATION = [[1, 0], [0, 1]]
NOISE_TRANSFORMATION = [[0, 0], [0, 0]]
OBSERVATION = [[1, 0], [0, 1]]
Q = [[0.25, 0], [0, 0.5]]
R = [[0.5, 0], [0, 0.75]]
P = [[1, 0], [0, 1]]

SYS = [STATE_TRANSFORMATION, NOISE_TRANSFORMATION, OBSERVATION]
COV = [Q, R]

test = KF.KalmanFilter(SYS, COV, P, INIT_STATE)
trueval = []

for k in range(100):
    test.predict()
    test.correct(np.random.normal([23.5, 36.5], [0.5, 0.25]))
    trueval.append(np.matrix([23.5, 36.5]))
test.show(trueval)
