#python 3.5.2

import numpy as np
from PIL import Image
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt 
	
def drawWindow(Y,img_width,img_height):
        img = Image.new('RGB', (img_width, img_height), "white")
        p = img.load()
        i=0
        for x in range(img_width):
             for y in range(img_height):
                   RGB_value = [int(Y[i][0]),int(Y[i][1]),int(Y[i][2])]
                   p[x,y] = tuple(RGB_value)
                   i+=1
        img.save('output.jpg')

img_input = input("Enter image name(without any extension):")
img = img_input + ".jpg"
im = Image.open(img)
img_width, img_height = im.size
px = im.load()

arr = []
for i in range(img_width):
	for j in range(img_height):
		pixel = px[i,j]
		arr.append(pixel)

X_initial = np.transpose(np.array(arr)) 

X = []
for i in range(len(X_initial)):
	a = []
	mean = np.mean(X_initial[i])
	for j in range(len(X_initial[i])):
		a.append(X_initial[i][j] - mean)
	X.append(a)

X = np.matrix(X)
X_transpose = np.transpose(X)

C_x = (X * X_transpose)/len(X_transpose)
print (C_x)

w, E_x = LA.eig(C_x)

Y = np.transpose(E_x) * np.matrix(X)
C_y = (Y * np.transpose(Y))/len(np.transpose(Y))

print (C_y)
Y = np.array(Y)
drawWindow(np.transpose(Y),img_width,img_height)


#Failure case
print ("Failure Case")
mean = np.zeros(2)
cov = np.array([[1,0],[0,1]]) # positive definite
X = np.transpose(np.random.multivariate_normal(mean, cov, int(1e4)))
plt.plot(X[0], X[1],'o')
plt.show()

N = len(X[0])
X = np.matrix(X)
C_x = (X * np.transpose(X))/N

_, E_x = LA.eig(C_x)
Y = np.transpose(E_x) * X

Y = np.array(Y)
plt.plot(Y[0], Y[1],'o')
plt.plot(E_x[0],'r','|')
plt.plot(E_x[0],'b','|')
plt.show()







	

