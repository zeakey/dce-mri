function res = computeGdConc(AA, BB, maxBase, tr, fa)

% if (nargin < 3)
%     BB = 1;
%     t10 = 1998;
% end
% 
% tr  = 3.89;
% fa  = 12;

rr1 = 3.9; % t1 relaxivity at 3T
[nx, ny, nz, tt] = size(AA);

mask = zeros(nx,ny,nz);
for zz = 1:nz
    mm = AA(:,:,zz,tt);
    maxmm = max(mm(:));
    mask(:,:,zz) = mm > maxmm*0.05;
end

C = zeros(nx, ny, nz, tt);
for zz = 1:nz
    for xx = 1:nx
        for yy = 1:ny
            if (mask(xx,yy,zz) == 1)
                gd_conc = zeros(tt,1);
                ss = squeeze(AA(xx,yy,zz,:));
                baseline = mean(ss(1:maxBase));
                
                if (length(BB(:)) == 1)
                    t10 = 1998;
                else
                    t10 = BB(xx,yy,zz); % 1998 msec
                end
                
                E10 = exp(-tr/t10);
                B   = (1-E10)/(1-cosd(fa)*E10);
                r10 = 1000.0/t10;
                
                for timeInd = maxBase+1:tt
                    
                    enhanced = ss(timeInd)/baseline;
                    if (enhanced > 0.8/B)
                        enhanced = 0.8/B;
                    end
                    
                    A = B*enhanced;
                    if (A > 0)
                        r1 = (-1000.0/tr)*log((1-A)/(1-cosd(fa)*A));
                        
                        if (r1 > r10)
                            gd_conc(timeInd) = (r1 - r10)/rr1;
                        end
                    end
                end
                C(xx,yy,zz,:) = gd_conc(:);
            end
            
        end
    end
    
end

res = C;
