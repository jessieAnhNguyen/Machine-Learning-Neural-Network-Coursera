# Machine-Learning-Neural-Network-Coursera
This project implements the backpropagation algorithm for Neural networks and apply it to the task of Hand-written digit recognition.
It is the programming exercise of Machine Learning course taught by Andrew Ng, Stanford University on Coursera.

The files that are already written by the assignment:
ex4.m - Octave/MATLAB script that steps through the exercise
ex4data1.mat - Training set of hand-written digits
ex4weights.mat - Neural network parameters 
submit.m - Submission script that sends solutions to servers
displayData.m - Function to help visualize the dataset
fmincg.m - Function minimization routine (similar to fminunc)
sigmoid.m - Sigmoid function
computeNumericalGradient.m - Numerically compute gradients
checkNNGradients.m - Function to help check your gradients
debugInitializeWeights.m - Function for initializing weights
predict.m - Neural network prediction function



The files that I write:
sigmoidGradient.m - Compute the gradient of the sigmoid function
randInitializeWeights.m - Randomly initialize weights
nnCostFunction.m - Neural network cost function
