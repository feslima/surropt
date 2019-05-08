# sqp active-set test

function sqp_test_octave
  x0 = [-1.8 1.7 1.9 -0.8 -0.8]';

  #[x, obj, info, iter, nf, lambda] = sqp(x0, @f, @g, [])
  
  x0 = [1,5,5,1]';
  lb = 1.0 * ones(4,1);
  ub = 5.0 * ones(4,1);

[x, obj, info, iter, nf, lambda] = sqp(x0, @objective, @hs71eq, @hs071ineq, lb, ub)
#hs71eq(x)
#hs071ineq(x)
 endfunction

function r = g(x)
  # equality constraint
  r = [sum(x.^2) - 10;
  x(2) * x(3) - 5*x(4)*x(5);
  x(1)^3 + x(2)^3 + 1];
endfunction

function obj = f(x)
  obj = exp(prod(x)) - 0.5*(x(1)^3 + x(2)^3 +1)^2;
endfunction

function r = hs071ineq(x)
  r = x(1)*x(2)*x(3)*x(4) - 25;
endfunction

function r = hs71eq(x)
  r = sum(x.^2) - 40;
endfunction

function obj = objective(x)
  obj = x(1)*x(4)*(x(1)+x(2)+x(3))+x(3);
endfunction