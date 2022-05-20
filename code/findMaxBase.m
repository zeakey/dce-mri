function res = findMaxBase(in)

[sx, sy, sl, tt] = size(in);
ind_x = (-20:20)+sx/2;
ind_y = (-20:20)+sy/2;

dum = zeros(sx, sy);
dum(ind_x, ind_y) = 1;
dummy = dum == 1;

tic = zeros(tt,1);
for ind_t = 1:tt
    im_tmp = in(:,:,round(sl/2),ind_t);
    tmp_val = mean(im_tmp(dummy));
    tic(ind_t,1) = tmp_val;
end

%figure; plot(tic,'r.-'); grid on;
tic_d = gradient(tic);
ind_max = find(tic_d == max(tic_d(:)));
ind_neg = find(tic_d(1:ind_max) < 0);

bnum = max(ind_neg) + 1;

baseline = mean(tic(1:bnum));

% If the next point is less than 1% higher than baseline, 
% increase the baseline number   
while(tic(bnum+1)/baseline < 1.01) 
    bnum = bnum + 1;
    baseline = mean(tic(1:bnum));
end

res = bnum;



