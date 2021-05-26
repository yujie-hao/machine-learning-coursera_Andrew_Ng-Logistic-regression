function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
sum = 0;
for i = 1:m
    h = getHyposHypothesis(theta, X, i);
    sum = sum + (-1 * log(h) * y(i) - (1 - y(i)) * log(1 - h));
end

sum2 = 0;
for j = 2:size(theta)
    sum2 = sum2 + theta(j)^2;
end

J = sum / m + sum2 * lambda / (2 * m);


% gradient
tempTheta = zeros(length(theta), 1);
% j = 1, same as costFunction
sum = 0;
for i = 1:m
    h = getHyposHypothesis(theta, X, i);
    sum = sum + (h - y(i)) * X(i, 1);
end
tempTheta(1) = sum / m;

% j >= 2
for j = 2:length(theta)
    sum = 0;
    for i = 1:m
        h = getHyposHypothesis(theta, X, i);
        sum = sum + (h - y(i)) * X(i, j);
    end
    tempTheta(j) = sum / m + lambda / m * theta(j);
end

grad = tempTheta;
end

% h
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