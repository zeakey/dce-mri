function [p_Tofts, ssd, res] = fit_Tofts(t, ct, aif, params, bb)

%--------------------------------------------------------------------------
% Fit to extended Tofts model, parameters [Ktrans kep t0]
%--------------------------------------------------------------------------
% disp('all ct is zero??????')
% disp(all(ct == 0))
if (nargin < 5)
    lb = [0.001, 0.001, 0];
    ub = [    1,    50, 1];
else
    lb = bb(:,1);
    ub = bb(:,2);
end

% constrained non-linear least squares fit
options = optimset('Display', 'off'); % supress output
if (nargin < 4)
    % initial guesses for parameters Ktrans, kep, toff
    a0 = [0.3, 1, 0];
    [params, ~] = lsqcurvefit(@fun_Tofts, a0, t, ct, lb, ub, options, aif);
    params(2) = a0(2);
end

[params, ssd, res] = lsqcurvefit(@fun_Tofts, params, t, ct, lb ,ub, options, aif);
p_Tofts = params;

