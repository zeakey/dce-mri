function res = run_test_new(ct, time, maxBase, mode)

[nx, ny, nz, ~] = size(ct);

% Assume initial AIF
tdel = 1/60;
tp   = (0:tdel:time(end));
tp   = tp(:);

% Assumed Parker AIF
Hct = 0.42;
cp_parker = parker_aif(0.809,0.330,0.17046,0.365,0.0563,0.132,1.050, ...
    0.1685,38.078,0.483,tp-ceil(time(maxBase)/tdel)*tdel)/(1-Hct);

aif_init = [tp cp_parker];

% Initial Guess
center_area = 6;
center_nx = ((-center_area+1):center_area)+nx/2;
center_ny = ((-center_area+1):center_area)+ny/2;
center_nz = round(nz/2);

% Averaged CA curve
cc = ct(center_nx,center_ny,center_nz,1:length(time));
cc = squeeze(mean(mean(mean(cc,1),2),3));

% FIT DATA To Kinetic Modeling
switch(mode)
    case 1
        [p_init, ssd] = fit_Tofts(time, cc, aif_init);
        % -------------------------------------------------
        % by Kai for debug
        cc_hat = fun_Tofts(p_init, time, aif_init);
        filename='../tmp/fit_cc.mat';
        if ~isfile(filename)
            save(filename, "cc", "cc_hat", "aif_init", "time", "p_init", "ssd");
        end
        % -------------------------------------------------
        p3_low = p_init(3)-0.1;
        if (p3_low < 0)
            p3_low = 0;
        end
        
        bb(:,1) = [0.001, 0.001, p3_low ]; % lower bounds
        bb(:,2) = [    5,     1, p_init(3)+0.25]; % upper bounds
    case 2
        [p_init, ~] = fit_Tofts_Disp(time, cc, aif_init);
        
        if ((p_init(3)-0.25) < 0)
            p3_low = 0;
        else
            p3_low = p_init(3)-0.25;
        end
        p3_high = p_init(3)+0.25;
        
        p4_low = p_init(4)-0.1;
        if (p4_low < 0)
            p4_low = 0;
        end
        
        bb(:,1) = [0.001, 0.001, p3_low,  p4_low ]; % lower bounds
        bb(:,2) = [    5,     1, p3_high, p_init(4)+0.25]; % upper bounds
    case 3
        [p_init, ~] = fit_Disp(time, cc, maxBase);
end

B    = zeros(nx,ny,nz,6);
time = time(:);

for zz = 1:nz
    disp(strcat(string(zz), " of ", string(nz)));
    for xx = 1:nx
        for yy = 1:ny
            
            ctTmp = squeeze(ct(xx,yy,zz,:));
            
            if (sum(ctTmp(:)) > 0)
                
                % FIT DATA To Kinetic Modeling
                switch(mode)
                    case 1
                        [p, ssd] = fit_Tofts(time, ctTmp, aif_init, p_init, bb);
                        
                        B(xx,yy,zz,1) = p(1); % Ktrans
                        B(xx,yy,zz,2) = p(2); % ve 
                        B(xx,yy,zz,3) = p(3); % tdel
                        B(xx,yy,zz,4) = ssd;  % residual
                    case 2
                        [p, ssd] = fit_Tofts_Disp(time, ctTmp, aif_init, p_init, bb);
                        
                        kappa_w = (p(3) - p3_low)./(p3_high - p3_low);
                        dum = kappa_w > 1;
                        kappa_w(dum) = 1;
                        dum = kappa_w < 0;
                        kappa_w(dum) = 0;
                        kappa_w = 1 - kappa_w;
                        
                        B(xx,yy,zz,1) = p(1); % Ktrans
                        B(xx,yy,zz,2) = p(2); % ve
                        B(xx,yy,zz,3) = p(3); % kappa
                        B(xx,yy,zz,4) = p(4); % tdel
                        B(xx,yy,zz,5) = ssd;  % residual
                        B(xx,yy,zz,6) = kappa_w;
                        
                    case 3
                        [p, ssd] = fit_Disp(time, ctTmp, maxBase, p_init);
                        
                        B(xx,yy,zz,1) = p(1); % beta
                        B(xx,yy,zz,2) = p(2); % kep
                        B(xx,yy,zz,3) = p(3); % kappa
                        B(xx,yy,zz,4) = p(4); % mu
                        B(xx,yy,zz,5) = p(5); % tdel
                        B(xx,yy,zz,6) = ssd;  % residual
                end
                
            end
        end
    end
    if (mod(zz,10) == 0)
        fprintf('z = %d\n',zz);
    end
end

res = B;

