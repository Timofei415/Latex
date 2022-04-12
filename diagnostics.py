from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

# sigmoid function
def sigmoid(x):
	return(1/(1+np.exp(-x)))
#the derivative of the sigmoid function
def sigmoidGradient(x):
	return(sigmoid(x)*(1-sigmoid(x)))

#function that transforms image into vector of length 400
def dataset(filename,examplesnumber):

	img = PIL.Image.open(filename) 
	          
	width, height = img.size
	data = []


	for i in range(examplesnumber):
		area = (20*(i%100), 20*(i//100),20+20*(i%100),20+20*(i//100))
		number = img.crop(area)
		imag = number.convert('RGB')

		vec = []
		for x in range(20):
			for y in range(20):
				R,G,B = imag.getpixel((x,y))
				vec.append(1-(R+G+B)/765)

		data.append(vec)


	data = np.array(data)
	return(data)

#finding cost for neural network with 1 hidden layer
def costFunction(X,y,Theta1,Theta2,num_lables,lamb):
	#calculating the value with given thetas
	m = len(X)
	X = np.concatenate((np.ones((m,1)), X), axis=1)
	a2 = sigmoid(np.matmul(X,Theta1.T))
	a2 = np.concatenate((np.ones((len(a2),1)), a2), axis=1)
	h = sigmoid(np.matmul(a2,Theta2.T))

	cost = np.zeros((m,1))

	#summing the cost for all lables via logistic cost function for neural network 
	for i in range(num_lables):
		hk = h[:,[i]]
		yk = y[:,[i]]
		cost = cost + (-np.multiply(yk,np.log(hk))-np.multiply((1-yk), np.log(1-hk)))

	#dividing total cost by the number of examples 
	J = np.sum(cost)/m

	#setting regression matricies with zeros for bias term
	Theta1Reg = np.concatenate((np.zeros((len(Theta1),1)), Theta1[:,1:1000]), axis=1)
	Theta2Reg = np.concatenate((np.zeros((len(Theta2),1)), Theta2[:,1:1000]), axis=1)

	#calculating the regression and adding it to the cost function
	reg = lamb*(np.sum(Theta1Reg**2) + np.sum(Theta2Reg**2))/(2*m)

	J = J + reg


	#setting the initial matricies for gradients
	Theta1_grad = np.zeros((len(Theta1),len(Theta1[0])))
	Theta2_grad = np.zeros((len(Theta2),len(Theta2[0])))


	#performing backpropagation to calculate delatas and gradients for every element in matrix
	for j in range(m):
		a1 = X[[j],:].T
		z2 = np.matmul(Theta1,a1)
		a2 = sigmoid(z2)
		a2 = np.concatenate((np.ones((1,1)),a2),axis=0)
		z3 = np.matmul(Theta2,a2)
		a3 = sigmoid(z3)


		delta3 = a3 - y[[j],:].T
		delta2 = np.multiply(np.matmul(Theta2.T,delta3)[1:1000,:], sigmoidGradient(z2))

		Theta1_grad = Theta1_grad + delta2*a1.T
		Theta2_grad = Theta2_grad + delta3*a2.T

	#calculating gradfients for both thetas with regression
	Theta1_grad = Theta1_grad/m + lamb*Theta1Reg/m
	Theta2_grad = Theta2_grad/m + lamb*Theta2Reg/m


	return(J,Theta1_grad,Theta2_grad)

#minimizing the cost by summing gradient values to thetas many times
def gradientDescent(X,y,Theta1,Theta2,num_lables,num_iter,alpha,lamb):
	for i in range(num_iter):
		temp = Theta1
		Theta1 = Theta1 - alpha*costFunction(X,y,Theta1,Theta2,num_lables,lamb)[1]
		Theta2 = Theta2 - alpha*costFunction(X,y,temp,Theta2,num_lables,lamb)[2]
		print('Itenation number', i, 'cost =', costFunction(X,y,Theta1,Theta2,num_lables,lamb)[0])

	return(Theta1,Theta2)


def main():

	#['0','1','2','3','4','5','6','7','8','9','x','y','n','k','e','+','-','=','(',')','>','<','f','d']
	#setting the number of examples and symbols used for training
	examplesnumber = 2400
	symbolnumber = 24
	#setting the outputs vector
	y = []
	for i in range(examplesnumber):
		row = symbolnumber*[0]
		row[i//100] = 1
		y.append(row)


	y0 = np.array(y)
		
	#creating training sets X, y and cross-validation sets X_val and y_val
	X0 = dataset('symbol_training_set.png',2400)

	Xval = []
	yval = []

	X = []
	y = []

	for i in range(len(X0)):
		if i % 100 > 80:
			Xval.append(X0[i])
			yval.append(y0[i])
		else:
			X.append(X0[i])
			y.append(y0[i])

	X = np.array(X)
	y = np.array(y)
	Xval = np.array(Xval)
	yval = np.array(yval)


	m = len(X)

	i = 300
	#setting lists of errors for training and cross-validation
	xaxis = []
	yaxistrain = []
	yaxisval = []

	while i < m:

		#performing training with i training examples 
		Theta1 = 0.12*2*np.random.rand(80,401) - 0.12
		Theta2 = 0.12*2*np.random.rand(24,81) - 0.12
		
		weights = gradientDescent(X[0:i],y[0:i],Theta1,Theta2,24,300,1,1)

		Theta1 = weights[0]
		Theta2 = weights[1]

		#calculating errors for training and validation sets
		error_train = costFunction(X[0:i],y[0:i],Theta1,Theta2,10,0)[0]

		error_val = costFunction(Xval,yval,Theta1,Theta2,10,0)[0]

		xaxis.append(i) 
		yaxistrain.append(error_train)
		yaxisval.append(error_val)

		i = i + 300

	#plotting the graph of errors 
	plt.plot(xaxis, yaxistrain, color='r', label= 'training error')
	plt.plot(xaxis, yaxisval, color='g', label= 'cross-validation error')
	plt.xlabel('examples')
	plt.ylabel('error')
	plt.legend()
	plt.show()

main()