function out = dispersed_aif(aif, beta) 

dt = aif(2,1)-aif(1,1);

% Vascular transport function h(t)
ht = 1/beta*exp(-aif(:,1)/beta);
ht = ht/(sum(ht(:))*dt);

% delayed and dispersed aif c_b^dispered(t)
aif_new = conv(aif(:,2),ht).*dt;

out = [aif(:,1) aif_new(1:length(aif))];
