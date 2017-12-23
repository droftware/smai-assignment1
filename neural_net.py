#Akshat Tandon 201503001

import numpy as np
import random
import math

def read_file(name):
	f = open(name, 'rU')
	numbersGrid = []
	numbersDigit = []
	currentNumber = []
	for line in f:
		line = line.strip()
		length = len(line)
		if length == 32 or length == 1:
			if length == 32:
				# print line
				currentNumber.append(line)
				# print currentNumber
			else:
				numbersGrid.append(currentNumber)
				numbersDigit.append(line)
				currentNumber = []
	sampledGrid = []
	currentNumber = []

	# for i in range(len(numbersDigit)):
	# 	print 'Number: ', numbersDigit[i]
	# 	for j in range(32):
	# 		print numbersGrid[i][j]
	# 	print 
	cnum = ''
	sampleNumbers = []
	for i in range(len(numbersDigit)):
		# print 'Number: ', numbersDigit[i]
		j = 0
		if numbersDigit[i] == '0' or numbersDigit[i] == '1' or numbersDigit[i] == '5':
			sampleNumbers.append(int(numbersDigit[i]))
			while j < 32:
				# currentNumber.append(numbersGrid[i][j][::4])
				cnum += numbersGrid[i][j][::4]
				# print numbersGrid[i][j][::4]
				j += 4

			cnumList = list(cnum)
			cnumList = [int(x) for x in cnumList]
			sampledGrid.append(cnumList)
			cnum = ''
			# print 'Cnum:',list(cnum)
			# currentNumber = []

	return (sampleNumbers, sampledGrid)

def sigmoid(x):
	ex = math.e ** x
	val = ex/(1 + ex)
	return val

def dif_sigmoid(x):
	sig = sigmoid(x)
	val = sig * (1 - sig)
	return val

def target_value(digit):
	if digit == 0:
		return np.array([0, 0])
	elif digit == 1:
		return np.array([0, 1])
	elif digit == 5:
		return np.array([1, 0])
	else:
		print 'ERROR: Num not found'

def binary_to_digit(z0, z1):
	if z0 == 0 and z1 == 0:
		return 0
	elif z0 == 0 and z1 == 1:
		return 1
	elif z0 == 1 and z1 == 0:
		return 5
	else:
		print 'ERROR: Not able to convert from binary: ',(z0, z1)


def feed_forward(x, wh, wo, nh, no):
	y = []
	z = []
	neth = []
	neto = []
	aug_x = np.append(1, x)
	for j in range(nh):
		net_j = aug_x.dot(wh[j, :])
		neth.append(net_j)
		y_j = sigmoid(net_j)
		y.append(y_j)


	yk = [1]
	yk = yk + y
	# y.insert(0, 1)
	aug_y = np.array(yk)

	for k in range(no):
		# print 'aug_y: ', aug_y
		# print 'wok: ', wo[k, :]
		net_k = aug_y.dot(wo[k, :])
		neto.append(net_k)
		# print 'net_k', net_k
		z_k = sigmoid(net_k)
		z.append(z_k)

	z = np.array(z)
	return z, y, neth, neto


def stochastic_backprop(digits, samples, d, nh, no, theta, eta):

	wh = np.random.rand(nh, d+1)
	wo = np.random.rand(no, nh+1)
	num_samples = len(samples)
	# print 'Num samples: ', num_samples
	count = 0
	delo = []
	delk = []

	while True:
		idx = random.randint(0, num_samples - 1)
		digit = digits[idx]
		t = target_value(digit)
		x = np.array(samples[idx])
		z, y, neth, neto = feed_forward(x, wh, wo, nh, no)
		aug_x = np.append(1, x)

		yk = [1]
		yk = yk + y
		# y.insert(0, 1)
		aug_y = np.array(yk)

		# x = aug_x
		for k in range(no):
			dif = t[k] - z[k]
			fdash = dif_sigmoid(neto[k])
			dell = dif * fdash
			delo.append(dell)
			for j in range(nh+1):
				delta = eta * dell * aug_y[j]
				wo[k, j] = wo[k, j] + delta

		for j in range(nh):
			fdash = dif_sigmoid(neth[j])
			sigma = 0
			for k in range(no):
				sigma += (wo[k, j] * delo[k])
			dell = fdash * sigma
			for i in range(d+1):
				delta = eta * dell * aug_x[i]
				wh[j, i] = wh[j, i] + delta

		z, y, neth, neto = feed_forward(x, wh, wo, nh, no)
		diff = t - z
		mod = np.linalg.norm(diff)
		j = 0.5 * mod * mod
		if j < theta:
			print j
			break
		count += 1

	print 'Count: ', count

	return wo,wh

def approx(x):
	if x< 0 or x >1:
		print 'ERROR: Value approximated greater than 1: ', x
	if x < 0.5:
		return 0
	else:
		return 1

def classify(sample, wh, wo, nh, no):
	x = np.array(sample)
	z, y, neth, neto = feed_forward(x, wh, wo, nh, no)
	# print 'Sample:',x
	# print 'z obtained:', z
	z0 = approx(z[0])
	z1 = approx(z[1])
	# print 'z approximated',(z0,z1)
	number = binary_to_digit(z0, z1)
	return number
	

def cross_validate(digits, samples, wh, wo, nh, no):
	length = len(digits)
	correct = 0
	for i in range(length):
		# print 'Number:', digits[i]
		number = classify(samples[i], wh, wo, nh, no)
		# print 'Classified: ',number
		# print 'Original Digit: ', digits[i]
		if number == digits[i]:
			# print 'Classified'
			correct += 1
		# else:
		# 	print 'Not classified'

	fr = correct * 1.0/length
	print 'Correctly classified:', fr * 100


def main():
	d = 64
	nh = 20
	no = 2
	theta = 0.002
	eta = 0.5

	digits, samples = read_file('optdigits-orig.tra')
	wo, wh = stochastic_backprop(digits, samples, d, nh, no, theta, eta)

	# qw = 5
	# number = classify(samples[qw], wh, wo, nh, no)
	# print 'Original Number:', digits[qw]
	# print 'Classified Num:', number
	# if number == digits[qw]:
	# 	print 'Classified Correctly'
	# else:
	# 	print 'NOT Classified'

	cvDigits, cvSamples = read_file('optdigits-orig.cv')

	cross_validate(cvDigits, cvSamples, wh, wo, nh, no)

	# print 'Hidden Unit weights'
	# for i in range(nh):
	# 	for j in range(65):
	# 		print '( '+ str(i) +', ' + str(j) + ') = ' + str(wh[i, j])

	# print 'Output Unit weights'
	# for i in range(no):
	# 	for j in range(nh+1):
	# 		print '( '+ str(i) +', ' + str(j) + ') = ' + str(wo[i, j])



if __name__ == "__main__":
	main()
