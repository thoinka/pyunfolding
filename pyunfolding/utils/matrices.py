import numpy as np


def diff2_matrix(n):
	return 2.0 * np.eye(n) - np.roll(np.eye(n), 1) - np.roll(np.eye(n), -1)


def diff1_matrix(n):
	return np.eye(n) - np.eye(n, k=1)