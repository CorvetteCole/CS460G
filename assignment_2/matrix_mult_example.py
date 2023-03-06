import numpy as np
import math
import random

def sigmoid(x):
	return 1/(1+math.pow(math.e,-x))
	
#The architecture is 4 - 3 - 1
#4x1
input = [1,2,3,4]
output = [-1]

#4x3
weights_h = []

#3x1
weights_o = []

for i in range(len(input)):
	weights_h.append(list(np.random.uniform(-1,1,3)))
	
for j in range(3):
	weights_o.append(random.uniform(-1,1))
#print(weights_h)
#print(weights_o)

#Forward pass
#Step 1, generate hidden outputs
bias = [-1,-1,-1]
h_in = np.dot(list(np.array(weights_h).transpose()), input)
print(h_in)
h_in = np.add(bias, h_in)
print(h_in)

h_out = list(map(sigmoid, h_in))
#print(h_in)
#print(h_out)

#Step 2, generate the full output
o_out = sigmoid(np.dot(weights_o, h_out))
#print(o_out)

#Forward pass complete!

#Backward pass
#Step 1, calculate deltas


#Step 2, propagate Deltas backward
#NOTE Not actually how to calculate delta at the output!
delta_o = 0.1
delta_h = np.dot(delta_o, weights_o)
#print(delta_h)
gradients_h = np.outer(delta_h, input)
#print(input)
#print(np.transpose(np.array(weights_h)))
#print(gradients_h)

multTest = [1,2,3,4]
multTest2 = [1,2,3,4]

print(np.multiply(multTest,multTest2))





