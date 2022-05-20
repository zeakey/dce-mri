function [p_Kety, ssd] = fit_Kety(t, ct, aif, params, bb)

%--------------------------------------------------------------------------
% Fit to extended Kety model, parameters [Ktrans ve vp t0]
%--------------------------------------------------------------------------

if (nargin < 5)
    lb = [0.001, 0.001, 0.001, 0];
    ub = [    5,     1,   0.3, 1];
else
    lb = bb(:,1);
    ub = bb(:,2);
end

% constrained non-linear least squares fit
options = optimset('Display', 'off'); % supress output

if (nargin < 4)
    % initial guesses for parameters Ktr, ve, vp, toff
    a0 = [0.3, 0.2, 0.2, 0];
    [params, ssd] = lsqcurvefit(@fun_Kety, a0, t, ct, lb, ub, options, aif);
    params(2) = a0(2);
end

[params, ssd] = lsqcurvefit(@fun_Kety, params, t, ct, lb, ub, options, aif);

p_Kety = params;

