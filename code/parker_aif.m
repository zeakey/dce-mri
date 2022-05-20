function cb = parker_aif(A1, A2, T1,  T2, sigma1, sigma2, alpha, beta, s, tau, time)

cb =   A1/(sigma1*sqrt(2*pi))*exp(-(time-T1).^2/(2*sigma1^2)) ...
     + A2/(sigma2*sqrt(2*pi))*exp(-(time-T2).^2/(2*sigma2^2)) ...
     + alpha*exp(-beta*time)./(1+exp(-s*(time-tau)));

dummy = time < 0;

cb(dummy) = 0;
end