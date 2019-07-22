function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

#Feedforward 
X = [ones(m,1) X];
z2 = X * (Theta1)';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2 * (Theta2)';

h = sigmoid(z3); #h(x)

#Modify the y values
y2 = zeros(m, num_labels);
for i=1:m
  y2(i, y(i,1)) = 1;
endfor

#Implement the cost function
for i=1:m
  J = J + (-1/m) * sum((log(h))(i, :) * (y2)'(:,i) + ((log(1 - h))(i, :) * (1 - y2)'(:, i)));
endfor

#Regularized the cost function
reg1 = 0;
for i = 1:size(Theta1, 1)
  for j = 2:size(Theta1,2)
    reg1 = reg1 + (Theta1(i,j) ^2);
  endfor
endfor

reg2 = 0;
for i = 1:size(Theta2, 1)
  for j = 2:size(Theta2,2)
    reg2 = reg2 + (Theta2(i,j) ^2);
  endfor
endfor

regularization = (lambda/(2*m)) * (reg1 + reg2);
J = J + regularization;

#Backpropagation algorithm
delta3 = zeros(num_labels, m);
delta3 = (h)' - (y2)';

z2 = [ones(m,1) z2];
delta2 = ((Theta2)' * delta3) .* (sigmoidGradient(z2))';
delta2 = delta2(2:end, :);

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

Delta1 = Delta1 + delta2 * X;
Delta2 = Delta2 + delta3 * a2;

Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2; 

#Regularized backpropagation
for i = 1:size(Theta1, 1)
  for j = 2:size(Theta1,2)
    Theta1_grad(i,j) = Theta1_grad(i,j) + (lambda/m) * Theta1(i,j);
  endfor
endfor


for i = 1:size(Theta2, 1)
  for j = 2:size(Theta2,2)
    Theta2_grad(i,j) = Theta2_grad(i,j) + (lambda/m) * Theta2(i,j);
  endfor
endfor




















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
