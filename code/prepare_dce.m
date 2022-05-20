clear all; clc; close all;

write_dicom = 1;
dr1 = '../dicom/';
dr2 = '../dicom_pk/';

filelist = dir(dr1);
filelist(ismember( {filelist.name}, {'.', '..', '.DS_Store'})) = [];  %remove . and ..
filelist = filelist([filelist.isdir]);

%%
for a1 = 1%1:numel(filelist)
    drtmp = [filelist(a1).folder '/' filelist(a1).name];
    flist2 = dir(drtmp);
    flist2(ismember( {flist2.name}, {'.', '..','.DS_Store'})) = [];  %remove . and ..
    
    % if there exist multiple exams for a subject
    for dd = 1:length(flist2)
        fullfn = fullfile(flist2(dd).folder,flist2(dd).name);
        flist3 =  dir(fullfn);
        flist3(ismember( {flist3.name}, {'.', '..','.DS_Store'})) = [];  %remove . and ..
        
        % find the folder containing DCE images
        dce_folder = {};
        multi_dcefile = 0;
        ind_tmp = 1;
        for bb = 1:length(flist3)
            nametmp = fullfile(flist3(bb).folder,flist3(bb).name);
            if contains(nametmp, 'iCAD-MCC_') || contains(nametmp,'DCAD-MCC-DYN') ...
                    || contains(nametmp,'t1_twist_tra') || contains(nametmp,'Twist_dynamic')
                
                if ~contains(nametmp, 'MIP') && ~contains(nametmp, 'Ktrans') && ~contains(nametmp, 'Kep')
                    dce_folder{ind_tmp} = nametmp;
                    ind_tmp = ind_tmp + 1;
                end
            end
        end
        if ~isempty(dce_folder)
            if length(dce_folder) > 1
                if length(dce_folder) == 2
                    if contains(dce_folder{1}, 'MCC') % if both MCC and non-MCC exist, select MCC
                        dce_folder_final = dce_folder{1};
                    else
                        dce_folder_final = dce_folder{2};
                    end
                else
                    dce_folder_final = dce_folder{1};
                    dce_folder_final = [dce_folder_final(1:end-11) '*'];
                    multi_dcefile = 1;
                end
            else
                dce_folder_final = dce_folder{1};
            end
        end
        
        % find the folder containing T10 images
        t10_folder = {};
        ind_tmp = 1;
        for bb = 1:length(flist3)
            nametmp = fullfile(flist3(bb).folder,flist3(bb).name);
            if contains (nametmp, 'iCAD-MCC-T10-Mapping') || contains(nametmp,'DCAD-MCC-T10-Mapping') ...
                    || contains (nametmp, 'iCAD-T10-Mapping') || contains(nametmp,'T1_Map_')
                t10_folder{ind_tmp} = nametmp;
                ind_tmp = ind_tmp + 1;
            end
        end
        if ~isempty(t10_folder)
            if length(t10_folder) == 2
                if contains(t10_folder{1}, 'MCC')
                    t10_folder_final = t10_folder{1};
                else
                    t10_folder_final = t10_folder{2};
                end
            else
                t10_folder_final = t10_folder{1};
            end
        end
        
        if ~isempty(t10_folder)
            flist_new =  dir(fullfile(t10_folder_final,'**/*.dcm'));
            im_tmp = dicomread(fullfile(flist_new(1).folder,flist_new(1).name));
            [sx, sy] = size(im_tmp);
            
            t10 = zeros(sx,sy,length(flist_new));
            for cc = 1:length(flist_new)
                nametmp = fullfile(flist_new(cc).folder,flist_new(cc).name);
                im_tmp = dicomread(nametmp);
                info_tmp = dicominfo(nametmp);
                t10(:,:,info_tmp.InstanceNumber) = im_tmp;
            end
            [sx, sy, sl] = size(t10);
        else
            sl = 20;
        end
        
        if ~isempty(dce_folder)
            flist_new =  dir(fullfile(dce_folder_final,'**/*.dcm'));
            nametmp = fullfile(flist_new(1).folder,flist_new(1).name);
            im_tmp = dicomread(nametmp);
            [sx, sy] = size(im_tmp);
            
            info_tmp = dicominfo(nametmp);
            tr = info_tmp.RepetitionTime;
            fa = info_tmp.FlipAngle;
            
            % sort out the DCE images
            time_tmp = zeros(length(flist_new),1);
            ind_all  = zeros(length(flist_new),1);
            for cc = 1:length(flist_new)
                nametmp = fullfile(flist_new(cc).folder,flist_new(cc).name);
                info_tmp = dicominfo(nametmp);
                
                % update time information
                acqTime = info_tmp.AcquisitionTime;
                t_tmp = 60.0*(60.0*str2double(acqTime(1:2))+str2double(acqTime(3:4)))+str2double(acqTime(5:end));
                time_tmp(cc,1) = t_tmp;
                
                % update instance number
                insNum = info_tmp.InstanceNumber;
                if (multi_dcefile == 1)
                    if isfield(info_tmp,'AcquisitionNumber')
                        acqNum = info_tmp.AcquisitionNumber;
                    elseif isfield(info_tmp,'SeriesNumber')
                        acqNum = info_tmp.SeriesNumber;
                    else
                        acqNum = 1;
                    end
                    indNum = (acqNum-1)*sl + insNum;
                else
                    indNum = insNum;
                end
                ind_all(cc,1) = indNum;
            end
            
            if min(ind_all(:)) ~= 1
                ind_all = ind_all - min(ind_all(:)) + 1;
            end
            
            % create DCE data
            dce = zeros(sx,sy,length(flist_new));
            time_all = zeros(length(flist_new),1);
            for cc = 1:length(flist_new)
                nametmp = fullfile(flist_new(cc).folder,flist_new(cc).name);
                im_tmp  = dicomread(nametmp);
                dce(:,:,ind_all(cc)) = im_tmp;
                time_all(ind_all(cc),1) = time_tmp(cc);
            end
            [sx, sy, sl_all] = size(dce);
        end
        
        % update time and reshape dce
        time_dce = time_all(1:sl:end);
        time_dce = time_dce - time_dce(1);
        dce = reshape(dce, sx, sy, sl, sl_all/sl);
        
        % find maxBase...
        maxBase = findMaxBase(dce);
        % fprintf('%d: %s (DCE - %d and T10 - %d) and MaxBase is %d \n',a1, filelist(a1).name, ~isempty(dce_folder), ~isempty(t10_folder), maxBase);
        
        % compute Gd concentration
        if ~isempty(t10_folder)
            dce_ct = computeGdConc(dce, t10, maxBase, tr, fa);
        else
            dce_ct = computeGdConc(dce, 1,   maxBase, tr, fa);
        end
        % compute PK map
        res_tofts = run_test_new(dce_ct, time_dce/60, maxBase, 1);
        
        %% Creating DICOM files
        if (write_dicom == 1)
            fullfn_new = strrep(fullfn,dr1,dr2);
            if(~exist(fullfn_new,'dir'))
                mkdir(fullfn_new);
            end
            
            if ~isempty(t10_folder)
                flist_new =  dir(fullfile(t10_folder_final,'**/*.dcm'));
                for cc = 1:length(flist_new)
                    nametmp = fullfile(flist_new(cc).folder,flist_new(cc).name);
                    
                    % Ktrans
                    dcmfileNew = [fullfn_new '/IM-0010-' num2str(cc,'%.4d'),'.dcm'];
                    info_tmp = dicominfo(nametmp);
                    im_tmp = res_tofts(:,:,info_tmp.InstanceNumber,1); % Ktrans
                    info_new = info_tmp;
                    info_new.SeriesNumber = 40000;
                    info_new.SeriesDescription = 'Ktrans-FA-0-E';
                    
                    dicomwrite(uint16(im_tmp*1000), dcmfileNew, info_new);
                    
                    % ve
                    dcmfileNew = [fullfn_new '/IM-0020-' num2str(cc,'%.4d'),'.dcm'];
                    info_tmp = dicominfo(nametmp);
                    im_tmp = res_tofts(:,:,info_tmp.InstanceNumber,2); % Ktrans
                    info_new = info_tmp;
                    info_new.SeriesNumber = 40001;
                    info_new.SeriesDescription = 've-FA-0-E';
                    
                    dicomwrite(uint16(im_tmp*1000), dcmfileNew, info_new);
                end
            else
                flist_new =  dir(fullfile(dce_folder_final,'**/*.dcm'));
                for cc = 1:sl
                    nametmp = fullfile(flist_new(cc).folder,flist_new(cc).name);
                    
                    % Ktrans
                    dcmfileNew = [fullfn_new '/IM-0010-' num2str(cc,'%.4d'),'.dcm'];
                    info_tmp = dicominfo(nametmp);
                    im_tmp = res_tofts(:,:,info_tmp.InstanceNumber,1); % Ktrans
                    info_new = info_tmp;
                    info_new.SeriesNumber = 40000;
                    info_new.SeriesDescription = 'Ktrans-FA-0-E';
                    
                    dicomwrite(uint16(im_tmp*1000), dcmfileNew, info_new);
                    
                    % ve
                    dcmfileNew = [fullfn_new '/IM-0020-' num2str(cc,'%.4d'),'.dcm'];
                    info_tmp = dicominfo(nametmp);
                    im_tmp = res_tofts(:,:,info_tmp.InstanceNumber,2); % Ktrans
                    info_new = info_tmp;
                    info_new.SeriesNumber = 40001;
                    info_new.SeriesDescription = 've-FA-0-E';
                    
                    dicomwrite(uint16(im_tmp*1000), dcmfileNew, info_new);
                end
            end
        end
    end
end
