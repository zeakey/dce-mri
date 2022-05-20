 function [p_Tofts, ssd, res] = fit_Tofts_Disp(t, ct, aif, params, bb)

%--------------------------------------------------------------------------
%   Fit to standard Tofts model with dispersion, 
%   parameters [Ktrans ve kappa t0]
%--------------------------------------------------------------------------

if (nargin < 5)
    lb = [0.001, 0.001, 0.0001, 0];
    ub = [    5,     1,    0.5, 0.5];
else
    lb = bb(:,1);
    ub = bb(:,2);
end

% constrained non-linear least squares fit
options = optimset('Display', 'off'); % supress output
if (nargin < 4)
    % initial guesses for parameters Ktr, ve, toff
    params = [0.3, 0.2, 0.1, 0];
%     [params, ~] = lsqcurvefit(@fun_Tofts_Disp, a0, t, ct, lb, ub, options, aif);
%     params(2) = a0(2);
end

[params, ssd, res] = lsqcurvefit(@fun_Tofts_Disp, params, t, ct, lb ,ub, options, aif);
p_Tofts = params;

