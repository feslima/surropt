function sqp_test_matlab
% octave
disp('---------octave------------')
x0 = [-1.8 1.7 1.9 -0.8 -0.8]';
options = optimoptions('fmincon','Algorithm','active-set');
[x,fval,exitflag,output,lambda] = fmincon(@f,x0,[],[],[],[],[],[],@g,options)

% hs071
disp('---------hs071------------')
x0 = [1,5,5,1]';
lb = 1.0 * ones(4,1);
ub = 5.0 * ones(4,1);
[x,fval,exitflag,output,lambda] = fmincon(@objective,x0,[],[],[],[],lb,ub,@nlcon,options)
lambda.eqnonlin
lambda.ineqnonlin

disp('---------rosenbrock_no_const------------')
x0 = [-1 2]';
[x,fval,exitflag,output,lambda] = fmincon(@rosenbrockwithgrad,x0,[],[],[],[],[],[],[],options)

disp('---------rosenbrock_1_ineq_const------------')
x0 = [-1 2]';
[x,fval,exitflag,output,lambda] = fmincon(@rosenbrockwithgrad,x0,[],[],[],[],[],[],@rosen_1con,options)
lambda.ineqnonlin
    function [f,g] = rosenbrockwithgrad(x)
        % Calculate objective f
        f = 100*(x(2) - x(1)^2)^2 + (1-x(1))^2;
        
        if nargout > 1 % gradient required
            g = [-400*(x(2)-x(1)^2)*x(1)-2*(1-x(1));
                200*(x(2)-x(1)^2)];
        end
    end

    function [c, ceq] = rosen_1con(x)
        c = x(1) + 2*x(2) - 1;
        ceq = [];
    end

    function [r, req] = g(x)
        r = [];
        % equality constraint
        req = [sum(x.^2) - 10;
            x(2) * x(3) - 5*x(4)*x(5);
            x(1)^3 + x(2)^3 + 1];
    end

    function obj = f(x)
        obj = exp(prod(x)) - 0.5*(x(1)^3 + x(2)^3 +1)^2;
    end

    function [c,ceq] = nlcon(x)
        c = 25.0 - x(1)*x(2)*x(3)*x(4);
        ceq = sum(x.^2) - 40;
    end

    function obj = objective(x)
        obj = x(1)*x(4)*(x(1)+x(2)+x(3))+x(3);
    end
end