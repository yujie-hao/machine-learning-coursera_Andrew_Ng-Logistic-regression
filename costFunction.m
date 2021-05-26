function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
sum = 0;
for i = 1:m
    h = getHyposHypothesis(theta, X, i);
    sum = sum + (-1 * log(h) * y(i) - (1 - y(i)) * log(1 - h));
end
J = sum / m;

% gradient
tempTheta = zeros(length(theta), 1);

for j = 1:length(theta)
    sum = 0;
    for i = 1:m
        h = getHyposHypothesis(theta, X, i);
        sum = sum + (h - y(i)) * X(i, j);
    end
    tempTheta(j) = sum / m;
end

grad = tempTheta;
end

function h = getHyposHypothesis(theta, X, i)
    z = 0;
    for j = 1:size(theta)
        % hθ(x) = θ0x0 + θ1x1 + ... + θnxn (already added x0 as x0 = 1, in ex2.mlx line 23)
        % matlab matrix index start from 1
        % Xi: means ith row, all columns (all features)    
        z = z + theta(j) * X(i, j);
    end
    h = sigmoid(z);
end