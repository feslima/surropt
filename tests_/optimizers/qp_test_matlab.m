% quadprog test
clear; clc;

H = [1 -1; -1 2];
A = [-1 0; 0 -1; 1 1; -1 2; 2 1];
b = [0 0 2 2 3]';
f = [-2 -6]';
x0 = [1 1]';

[x,fval,exitflag,output,lambda] = quadprog(H,f,A,b,[],[],[],[])
lambda.ineqlin
lambda.lower
lambda.upper

H = eye(4);
f = [12 1 2 11]';
Aeq = [2 10 10 2];
beq = 12;
A = [-25 -5 -5 -25; -eye(4); eye(4)];
b = [0 0 -4 -4 0 -4 0 0 -4]';
[x,fval,exitflag,output,lambda] = quadprog(H,f,A,-b,Aeq,-beq,[],[])