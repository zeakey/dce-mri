function F = fun_MRDI(a,time)
%
% a model parameters [alpha kep kappa mu tdel];
% t time
% aif : evenly sampled aif, 1st col time, 2nd col Cp (time not nec same as t if aif calculated from fit)
%
tdel = 1/60;
tp = [tdel:tdel:time(end)]';
cp = sqrt(a(3)./(2*pi*tp)).*exp(-a(3)./(2*tp).*(tp-a(4)).^2);

%calculate the impulse response function at the resolution of the aif
imp = a(1).*exp(-a(2)*tp);

%Convolve impulse response with AIF and multiply by dt
convolution = conv(cp,imp).*tdel;

f = convolution(1:length(cp));

% interpolate at timepoints of uptake curve and include offset time
% as fitting parameter
F = interp1(tp,f,time-a(5),'pchip',0);