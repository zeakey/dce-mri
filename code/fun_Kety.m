function F = fun_Kety(a,t,aif)

% a model parameters [Ktrans ve vp t0]
% t time
% aif : evenly sampled aif, 1st col time, 2nd col Cp (time not nec same as t if aif calculated from fit)

%calculate the aif timestep
dt = aif(2,1)-aif(1,1);

%calculate the impulse response function at the resolution of the aif
imp = a(1).*exp(-a(1)*aif(:,1)./a(2));

%Convolve impulse response with AIF and multiply by dt
convolution = conv(aif(:,2),imp).*dt;

f = a(3).*aif(:,2) + convolution(1:length(aif));

% interpolate at timepoints of uptake curve and include offset time
% as fitting parameter
F = interp1(aif(:,1),f,t-a(4),'pchip',0);
