function F = fun_Tofts_Disp(a,t,aif)

% a model parameters [Ktrans ve kappa t0]
% t time
% aif : evenly sampled aif, 1st col time, 2nd col Cp (time not nec same as t if aif calculated from fit)

% calculate the aif timestep
dt = aif(2,1)-aif(1,1);

% calculate the impulse response function at the resolution of the aif
imp = a(1).*exp(-a(1)*aif(:,1)./a(2));

% delayed and dispersed aif
ht = 1/a(3)*exp(-aif(:,1)/a(3));
ht = ht/(sum(ht(:))*dt);

aif_new = conv(aif(:,2),ht).*dt;
aif_new = aif_new(1:length(aif));

%Convolve impulse response with AIF and multiply by dt
convolution = conv(aif_new,imp).*dt;

f = convolution(1:length(aif));

% interpolate at timepoints of uptake curve and include offset time
% as fitting parameter
F = interp1(aif(:,1),f,t-a(4),'pchip',0);
