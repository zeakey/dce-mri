function [p_MRDI, ssd, res] = fit_MRDI(time, ct, maxBase, a0)

%--------------------------------------------------------------------------
% Fit to MRDI model, parameters [alpha kep kappa tau tdel]
%--------------------------------------------------------------------------
% For each t0 in a selected 20 second interval, with a resolution of 3.1 seconds, 
% the trust-region reflective search tool is performed. 
% The chosen search interval (trust region) 
% for k = [0.004, 4]s-1 and kep = [0.01, 10]min-1.
% --- Investigative Radiology & Volume 49, Number 8, August 2014
%
% a = [alpha kep kappa mu tdel];
%

% Initial condition...
if (nargin < 4)
    a0 = [3, 0.3, 1*60, 10/60, time(maxBase)];
end

if (time(maxBase)-time(1) > 0)
    lbt = time(maxBase)-time(2);
else
    lbt = 0;
end

lb = [0.001, 0.001, 0.004*60,  5/60, lbt];
ub = [    2,     1,     1*60, 60/60, time(maxBase)+time(2)];

% constrained non-linear least squares fit
options = optimset('Display', 'off'); % supress output

[params, ssd, res] = lsqcurvefit(@fun_MRDI, a0, time(:), ct(:), lb, ub, options);

p_MRDI = params;


