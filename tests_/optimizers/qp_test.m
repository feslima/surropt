# quadprog example test
H = [1 -1; -1 2];
f = [-2 -6]';
A = [1 1; -1 2; 2 1];
Aub = [2 2 3]';
Alb = -Inf(size(Aub));
lb = zeros(2, 1);
ub = Inf(size(lb));

x0 = [1 1]';

[x, obj, info, lambda] = qp(x0, H, f, [], [], lb, , Alb, A, Aub)