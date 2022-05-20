function F = fun_Tofts(a,t,aif)

% a model parameters [Ktrans kep t0]
% t time
% aif : evenly sampled aif, 1st col time, 2nd col Cp (time not nec same as t if aif calculated from fit)

%calculate the aif timestep
dt = aif(2,1)-aif(1,1);

%calculate the impulse response function at the resolution of the aif
imp = a(1).*exp(-a(2)*aif(:,1));

%Convolve impulse response with AIF and multiply by dt
convolution = conv(aif(:,2),imp).*dt;

f = convolution(1:length(aif));

% interpolate at timepoints of uptake curve and include offset time
% as fitting parameter
F = interp1(aif(:,1),f,t-a(3),'pchip',0); %t - a(3)+.2


