% quadprog test
H = [1 -1; -1 2];
A = [-1 0; 0 -1; 1 1; -1 2; 2 1];
b = [0 0 2 -5 3]';
f = [-2 -6]';
x0 = [1 1]';

[x,fval,exitflag,output,lambda] = quadprog(H,f,A,b,[],[],[],[],[1 1]')