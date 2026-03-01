"""
#X = np.array([[1, 2, 3, 2.5],
#              [2.0, 5.0, -1.0, 2.0],
#              [-1.5, 2.7, 3.3, -0.8],
#              [3.0, -1.0, 2.0, 1.0],
#              [2.0, 0.5, -1.5, 2.0],
#              [-1.0, 2.0, 1.0, -1.5]])
#y = np.array([0, 1, 2, 3, 0, 1])




Neurontesting (basic)
"""
"""
weights = [[2.0, 3.0, 2.0, 2.0], 
           [0.5, -0.91, -0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]
bias = [2, 3, 0.5]

weights2 = [[2.0, 3.0, 2.0, 2.0], 
           [0.5, -0.91, -0.26, -0.5], 
           [-0.26, -0.27, 0.17, 0.87]]
bias2 = [2, 3, 0.5]


layer1Outputs = np.dot(X, np.array(weights).T) + bias
print(output)
"""

"""
# "n_" is for neuron_
#Nueronlayer
layer_output2 = [] #layeroutput
for N_weights, n_bias in zip(weights, bias):
    #single Neuron
    neuron_output = 0 #neuronoutput
    for n_input, n_weight in zip(inputs, N_weights):
        neuron_output += n_input * n_weight
    neuron_output += n_bias
    layer_output2.append(neuron_output)

print(layer_output2)



#ActivationReLu -> fast training
#function (cut negative off)
inputs = [0.2, 0.4, 0.6, -5, 20]
output = []
for i in inputs:
    '''
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)
    '''
    output.append(max(0, i))
"""

"""
#ActivationSoftMax or Sigmoid called -> very slow training
 = 1 / (1 + e**-x)

#for inputRow in input:
expValues = [] # = np.exp(input)
for output in input:
    expValues.append(E ** output)


normBase = sum(expValues, axis=1)
normValues = [] # = expValues / np.sum(expValues)

for value in expValues:
    normValues.append(value / normBase)

-->
expValues = np.exp(input)

normValues = expValues / np.sum(expValues, axis=1, keepdims = True)
"""




"""
#try1

# categorial cross etropy
# -> calculating loss / Cost

# problem log off 0

# One-Hot Encoding:
#   classes (how many indexes there are), 
#   label (which is hot), 
#   One-Hot[0,1,0,0] (every index is 0, except the hot one is 1)
#
#   -> in batch there is only the interesting index in the classTarget(before: targetOutput)
#      not every 0

# single one
#input  (softmax output)
targetOutput = [1, 0, 0]

loss = - math.log(input[0])

# batch
classTarget = [0, 1, 1]

for targIDX, distribution in zip(classTarget, input):
    distribution[targIDX]

    

#try2

math.sum((wertN - erwartungN)**2)
"""


# backpropagation = computes gradient of los-function
# step 2 use this gradient to optimize the network ()

# backpropagation
# Stochastic gradient descent