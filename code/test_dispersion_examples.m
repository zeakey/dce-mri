clear all; clc; close all;
addpath(strcat(pwd,'/gaussian_mixture_model'));

for case_num = 1
    
    switch(case_num)
        case 1
            casename = 'rkm9008'; % Great case
            maxBase = 7;
            sl = 11;
            L = [77 91]; % GS 3+4
            TZ = [70 78];
            PZ = [77 75];
            
        case 2
            casename = 'sm6395'; % Great case
            maxBase = 8;
            sl = 13; % GS 3+4
            L = [74 75];
            TZ = [79 86];
            PZ = [88 85];
        case 3
            casename = 'sm6395'; % Great case
            maxBase = 8;
            sl = 14; % FP
            L = [86 67];
            TZ = [79 86];
            PZ = [88 83];
    end
    
    fdir = ['data/' casename];
    fname = ['/dce_' casename '3d.mat'];
    load([fdir fname]);
    load([fdir '/time_dce.txt']);
    
    [nx_initial, ny_initial, nz_initial, tt] = size(ct);
    time = time_dce/60;
    
    % Assume initial AIF
    tdel = 1/60;
    tp   = (0:tdel:time(end));
    tp   = tp(:);
    
    % Assumed Hct
    Hct = 0.42;
    
    %Parker AIF
    cp_parker = parker_aif(0.809,0.330,0.17046,0.365,0.0563,0.132,1.050, ...
        0.1685,38.078,0.483,tp-ceil(time(maxBase)/tdel)*tdel)/(1-Hct);
    
    % Weinmann AIF
    cp_weinmann = biexp_aif(3.99, 4.78, 0.144, 0.011, tp-ceil(time(maxBase)/tdel)*tdel)/(1-Hct);
    
    % Fritz-Hans AIF
    cp_fh = biexp_aif(24, 6.2, 3.0, 0.016, tp-ceil(time(maxBase)/tdel)*tdel)/(1-Hct); % Fritz-Hans
    
    aif1 = [tp cp_parker];
    aif2 = [tp cp_weinmann];
    aif3 = [tp cp_fh];
    
    %%
    res_tofts = zeros(3,3);
    res_disp = zeros(4,3);
    
    figure(case_num);
    for abc = 1:3
        switch(abc)
            case 1
                ll = L;
                name = 'Lesion';
            case 2
                ll = TZ;
                name = 'Transition Zone';
            case 3
                ll = PZ;
                name = 'Peripheral Zone';
        end
        
        % Do not compute PK maps if CA uptake is too small..
        cc_init = squeeze(ct(ll(1),ll(2),sl,:));
        
        bb(:,1) = [0.001, 0.001,    0];     % lower bounds
        bb(:,2) = [    1,     1, 0.25];     % upper bounds
        initial_parameter1 =  [0.2, 0.2, 0];
        
        % FIT DATA To Tofts Modeling
        [p_init1, ssd1, res1] = fit_Tofts(time, cc_init, aif1, initial_parameter1, bb);
        e1 = sum(res1(maxBase+1:tt).^2)*100;
        
        [p_init2, ssd2, res2] = fit_Tofts(time, cc_init, aif2, initial_parameter1, bb);
        e2 = sum(res1(maxBase+1:tt).^2)*100;

        [p_init3, ssd3, res3] = fit_Tofts(time, cc_init, aif3, initial_parameter1, bb);
        e3 = sum(res1(maxBase+1:tt).^2)*100;
        
        res_tofts(:,abc) = p_init1;
        
        bb_disp(:,1) = [0.001, 0.001,    0,    0];      % lower bounds
        bb_disp(:,2) = [    5,     1, 0.35, 0.25];      % upper bounds
        initial_parameter_disp =  [0.3, 0.2, 0.1, 0];
        
        beta_all = linspace(0,0.5,10);
        beta_all(1) = beta_all(2)/2;
        aif_dispersed = zeros(length(tp),2,length(beta_all));
        p_init_tmp = zeros(3,length(beta_all));
        res_tmp = zeros(tt,length(beta_all));
        
        % Create dispersed AIF based on various beta
        for ind = 1:length(beta_all)
            beta = beta_all(ind);
            aif_dispersed(:,:,ind) = dispersed_aif(aif1,beta);
        end
        
        for ind = 1:length(beta_all)
            if ind == 1
                p_init_tmp(:,ind) = p_init1;
                res_tmp(:,ind) = res1;
            else
                [p_init, ssd, res] = fit_Tofts(time, cc_init, aif_dispersed(:,:,ind), initial_parameter1, bb);
                p_init_tmp(:,ind) = p_init;
                res_tmp(:,ind) = res;
            end
        end
        
        error_tmp = sum(res_tmp(maxBase+1:tt,:).^2)*100;
        min_ind = find(error_tmp == min(error_tmp(:)),1);
        res_disp(1:3,abc) = p_init_tmp(:,min_ind);
        res_disp(4,abc) = beta_all(min_ind);
        
        %% Plot
        fitted_curve1 = fun_Tofts(p_init1(:), time, aif1);
        fitted_curve2 = fun_Tofts(p_init2(:), time, aif2);
        fitted_curve3 = fun_Tofts(p_init3(:), time, aif3);
        fitted_curve4 = fun_Tofts(p_init_tmp(:,min_ind), time, aif_dispersed(:,:,min_ind));
        
        subplot(2,2,abc); plot(time, cc_init, 'r.-', time, fitted_curve1, 'g', time, fitted_curve4, 'b'); grid on;
        title(name);
        
        fprintf('Fitting errors for %s are %.2f and %.2f (/w Disp of %.2f)\n',name, e1, min(error_tmp(:)), beta_all(min_ind));
        
    end
    fprintf('Done with Case: %s\n',casename);
    
end

