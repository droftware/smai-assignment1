#Akshat Tandon 201503001

import numpy as np
import matplotlib.pyplot as plt


def normalize(x):
	return -1 * x

def augment(x):
	numRows = x.shape[0]
	onesVec = np.ones((numRows, 1))
	y = np.append(onesVec, x, 1)
	return y

def lms(y, a, b, eta, theta):
	k = 0
	n = y.shape[0]
	d = y.shape[1]
	classified = 0
	count = 0
	etaOne = eta
	while True:
		factor = b - a.dot(y[k, :])
		# print 'Factor', factor
		additive = eta * factor * y[k, :]
		# print 'Additive', additive
		a = a + additive
		# print 'Mod additive:', np.linalg.norm(additive)
		if np.linalg.norm(additive) < theta:
			break
		k = (k + 1) % n
		count += 1
		eta = etaOne/count
		# print 'Count:', count
	print 'lms Count:', count
	return a


def relaxation_margin(y, a, b, eta):
	k = 0
	n = y.shape[0]
	d = y.shape[1]
	classified = 0
	count = 0
	while True:
		if a.dot(y[k, :]) <= b:
			# print 'Misclassified:', y[k, :]
			# print
			# print 'a: ',a
			# print
			magnitude = np.linalg.norm(y[k, :])
			magnitude2 = magnitude * magnitude
			factor = (b - a.dot(y[k, :])/magnitude2)
			a = a + eta * factor * y[k, :]
			classified = 0
		else:
			# print 'Classified:', y[k, :]
			classified += 1
		k = (k + 1) % n
		if classified == n:
			break
		count += 1
	print 'relaxation_margin Count:', count
	return a

def perceptron_margin(y, a, b):
	k = 0
	n = y.shape[0]
	d = y.shape[1]
	classified = 0
	count = 0
	while True:
		if a.dot(y[k, :]) <= b:
			# print 'Misclassified:', y[k, :]
			# print
			# print 'a: ',a
			# print
			a = a + y[k, :]
			classified = 0
		else:
			# print 'Classified:', y[k, :]
			classified += 1
		k = (k + 1) % n
		if classified == n:
			break
		count += 1
	print 'perceptron_margin Count:', count
	return a


def perceptron(y, a):
	k = 0
	n = y.shape[0]
	d = y.shape[1]
	classified = 0
	count = 0
	while True:
		if a.dot(y[k, :]) <= 0:
			# print 'Misclassified:', y[k, :]
			# print
			# print 'a: ',a
			# print
			a = a + y[k, :]
			classified = 0
		else:
			# print 'Classified:', y[k, :]
			classified += 1
		k = (k + 1) % n
		if classified == n:
			break
		count += 1
	print 'perceptron Count:', count
	return a

def plotPoints(w1, w2):
	w1y1 = w1[:, 0]
	w1y2 = w1[:, 1]

	# print 'w1y1',w1y1
	# print 'w1y2',w1y2

	w2y1 = w2[:, 0]
	w2y2 = w2[:, 1]

	plt.scatter(w1y1, w1y2, color='red')
	plt.scatter(w2y1, w2y2, color='blue')


def plotDiscriminant(a):
	xq = np.array([0, 10])
	yq = (-a[0] - a[1]*xq)/a[2]
	plt.plot(xq, yq)

def plotShow():
	plt.xlabel('x1 axis')
	plt.ylabel('x2 axis')
	plt.title('Linear Discriminant Functions')
	plt.legend(['single sample perceptron', 'single sample perceptron with margin', 'relaxation algorithm with margin', 'widrow-hoff'])
	# plt.legend(['single sample perceptron with margin', 'relaxation algorithm with margin'])
	# plt.legend(['widrow hoff'])
	plt.show()


def main():
	w1 = np.array([[2, 7], [8, 1], [7, 5], [6, 3], [7, 8], [5, 9], [4, 5]])
	w2 = np.array([[4, 2], [-1, -1], [1, 3], [3, -2], [5, 3.25], [2, 4], [7, 1]])

	# w1 = np.array([[2, 7], [8, 1], [7, 5], [6, 3], [7, 8], [5, 9], [4, 5], [4, 1], [3, -1]])
	# w2 = np.array([[4, 2], [-1, -1], [1, 3], [3, -2], [5, 3.25], [2, 4], [7, 1]])

	# x = np.append(w1, w2, 0)
	# print w2*-1
	# temp = normalize(augment(w2))

	y1 = augment(w1)
	y2 = augment(w2)
	plotPoints(w1, w2)
	# print 'y1',y1
	nor_y2 = normalize(y2)
	# print 'nor y2', nor_y2
	y = np.append(y1, nor_y2, 0)

	d = y.shape[1]
	a_init = np.random.rand((d))
	# a_init = np.ones((d))
	# a_init = np.array([-85, 7, 9])
	# a_init = np.array([-89, 11, 10])

	print 'a- init:', a_init
	b = 0.2
	eta = 0.7
	theta = 0.7

	a = perceptron(y, a_init)
	plotDiscriminant(a)

	
	a = perceptron_margin(y, a_init, b)
	plotDiscriminant(a)


	
	a = relaxation_margin(y, a_init, b, eta)
	plotDiscriminant(a)

	
	a = lms(y, a_init, b, eta, theta)
	plotDiscriminant(a)

	print 'Perceptron is: ', a

	plotShow()



	# plotPoints(w1, w2)
	# plotDiscriminant(a)




if __name__ == "__main__":
	main()
