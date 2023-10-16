import numpy as np

def GetRotationMatrix(theta):
	R = np.array([
				[np.cos(theta), -np.sin(theta)],
				[np.sin(theta),  np.cos(theta)]
				])

	return R

def GetAngle(p1, p2):
	return np.arctan2((p1[1] - p2[1]), (p1[0] - p2[0]))


def GetTransformationMatrix(theta, x, y):
	R = GetRotationMatrix(theta)
	T = np.append(R, np.array([[x, y]]).T, axis = 1)
	T = np.append(T, np.array([[0, 0, 1]]), axis = 0)

	return T


def GetInverseMatrix(M):
	return np.linalg.inv(M)

def AlmostEqual(p1, p2, thresh = 1e-4):
	d = np.sqrt((p1[0, 0] - p2[0, 0])**2 + (p1[1, 0] - p2[1, 0])**2)

	if d<thresh:
		return True
	else:
		return False


gravity = -5
TIME_STEP = 0.001