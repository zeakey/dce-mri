function cb = biexp_aif(a1, a2, m1, m2, time)

cb =   a1*exp(-m1*time) + a2*exp(-m2*time);

dummy = time < 0;
cb(dummy) = 0;