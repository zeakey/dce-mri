function res = run_test(ct, time, maxBase, mode)

[nx, ny, nz, tt] = size(ct);

% Assume AIF
Hct = 0.42;
tdel = 1/60;
tp = [0:tdel:time(end)]';
cp = parker_aif(0.809,0.330,0.17046,0.365,0.0563,0.132,1.050, ...
    0.1685,38.078,0.483,tp-ceil(time(maxBase)/tdel)*tdel)/(1-Hct);

aif_new = [tp cp];

switch(mode)
    case 1
        a0 = [0.3, 0.2, 0, 0, 0, 0];
    case 2
        a0 = [0.3, 0.2, 0.2, 0, 0, 0];
    case 3
        a0 = [0.3, 0.3, 0.3, 0.1, 0, 0];
    case 4
        a0 = [0.3, 0.3, 0.3, 0.1, time(maxBase-2), 0];
end
maxFactor = 2;
Bprev = repmat(a0(:),[1 nx/(2^(maxFactor+1)) ny/(2^(maxFactor+1)) nz]);
Bprev = permute(Bprev,[2 3 4 1]);

factor_all = [maxFactor:-1:0];

for abc = factor_all;
    factor = 2^abc;
    
    B    = zeros(nx/factor,ny/factor,nz,6);
    Init = zeros(nx/factor,ny/factor,nz,6);
    
    if (sum(size(Init)) ~= sum(size(Bprev)))
        Init(1:2:end,1:2:end, :, :) = Bprev;
        Init(2:2:end,1:2:end, :, :) = Bprev;
        Init(1:2:end,2:2:end, :, :) = Bprev;
        Init(2:2:end,2:2:end, :, :) = Bprev;
    else
        Init = Bprev;
    end
    
    ct_new = adjust_image(ct, factor, factor, 1);
    
    for zz = 1:nz
        for xx = 1:nx/factor
            for yy = 1:ny/factor
                
                InitTmp = squeeze(Init(xx,yy,zz,:));
                ctTmp = squeeze(ct_new(xx,yy,zz,:));
                if (sum(ctTmp(:)) > 0)
                    % FIT DATA TO TOFTS MODEL
                    switch(mode)
                        case 1
                            [p, ssd] = fit_Tofts(time(:), ctTmp(:), aif_new, InitTmp(1:3));
                            
                            B(xx,yy,zz,1) = p(1); % Ktrans
                            B(xx,yy,zz,2) = p(2); % ve
                            B(xx,yy,zz,3) = p(3); % tdel
                            B(xx,yy,zz,4) = ssd;  % residual
                        case 2
                            [p, ssd] = fit_Kety(time(:), ctTmp(:), aif_new, InitTmp(1:4));
                            
                            B(xx,yy,zz,1) = p(1); % Ktrans
                            B(xx,yy,zz,2) = p(2); % ve
                            B(xx,yy,zz,3) = p(3); % vp
                            B(xx,yy,zz,4) = p(4); % tdel
                            B(xx,yy,zz,5) = ssd;  % residual
                        case 3
                            [p, ssd] = fit_aath(time(:), ctTmp(:), aif_new, InitTmp(1:5));
                            
                            B(xx,yy,zz,1) = p(1); % E
                            B(xx,yy,zz,2) = p(2); % Frho
                            B(xx,yy,zz,3) = p(3); % ve
                            B(xx,yy,zz,4) = p(4); % tau
                            B(xx,yy,zz,5) = p(5); % tdel
                            B(xx,yy,zz,6) = ssd;  % residual
                        case 4
                            [p, ssd] = fit_Disp(time(:), ctTmp(:), maxBase, InitTmp(1:5));
                            
                            B(xx,yy,zz,1) = p(1); % alpha
                            B(xx,yy,zz,2) = p(2); % kep
                            B(xx,yy,zz,3) = p(3); % kappa
                            B(xx,yy,zz,4) = p(4); % mu
                            B(xx,yy,zz,5) = p(5); % tdel
                            B(xx,yy,zz,6) = ssd;  % residual
                    end
                end
                
            end
            if (mod(xx,10) == 0)
                fprintf('x = %d\n',xx);
            end
            
        end
    end
    Bprev = B;
    
    fprintf('Multi-resolution Analysis: Factor = %d\n',factor);
    figure;
    subplot(2,3,1); imshow(abs(B(:,:,1,1)),[0 1]); title('p(1)');
    subplot(2,3,2); imshow(abs(B(:,:,1,2)),[0 1]); title('p(2)');
    subplot(2,3,3); imshow(abs(B(:,:,1,3)),[0 100]); title('p(3)');
    subplot(2,3,4); imshow(abs(B(:,:,1,4)),[0 1]); title('p(4)');
    subplot(2,3,5); imshow(abs(B(:,:,1,5)),[0 1]); title('p(5)');
    subplot(2,3,6); imshow(abs(B(:,:,1,6)),[0 1]); title('p(6)');
    colormap('jet'); pause(0.01);
end

res = B;

