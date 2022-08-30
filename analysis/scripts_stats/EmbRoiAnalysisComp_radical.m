
%% Initialize variables
    
clear
close all
% addpath(genpath('/path/to/BrainStat/brainstat_matlab/'));
addpath(genpath('./surfstat/'));
addpath(genpath('./export_fig-master'));
addpath(genpath('/home/romy.beaute/projects/hypnomed/softwares/surfstat/'));

%%load TPJ mask 
% mask_path = "G:\CosmoMVPA\scripts\data\TPJ_mask";
% mask_path = "DATA"
% R_TPJ =  niftiread(mask_path+"\fs5.R.func.nii");
% L_TPJ = niftiread(mask_path+"\fs5.L.func.nii");
% TPJ_mask = cat(1,L_TPJ,R_TPJ);
% TPJ_mask = TPJ_mask~=0;



p.analysis_framework = 'Daniel';
contrast_id = 1; %don't forget to set up contrast
%Contrast used States : [2,3,4,5] ;;; Nov vs Exp :[14,15,16,17] ;;;
%Exp_states :[6,7,8,9] ;;; Nov_states :[10,11,12,13];;; Interaction :[18,19,20,21]
% p.states_wanted = {'OP','RS'};
p.states_wanted = {'control','meditation'};
p.interaction = 0; %state_id to be removed from other state e.g. : OP-RS p.interaction = 3
% p.groups_wanted = {'all'} %{'experts','experts'};
%Define states to be compared
% p.prefixe = 'Mixed_RS_exp_vs_OP_nov';
p.prefixe = 'control_vs_meditation';

%exclude subjs 73 
p.outliers = [27,32]; %subject number that you want to exclude from the analysis
p.clusp = 0.05; %Set the clusters p-value 
p.col_lim_dim = [-4 4]; %Set max values for the figures
p.diff_lim = [-1 1]; %Set max values for the interaction figures
p.dims_wanted = [1];


% if strcmp(p.analysis_framework,'Daniel')
% d.emb = load('./control_meditation_hypnose_embedding.mat');
d.emb = load('/home/romy.beaute/projects/hypnomed/diffusion_embedding/emb_matrices/group_control_meditation_hypnose_embedding.mat');
% else
%     d.emb = load('./co_om_rs_group_embedding_GCCA.mat');
% end


d.emb = d.emb.emb;

surfaces_path = '/home/romy.beaute/projects/hypnomed/data/template/fsaverage/surf';
% d.surf.inflated = SurfStatReadSurf({['./lh.inflated'],['./rh.inflated']});
d.surf.inflated = SurfStatReadSurf({['/home/romy.beaute/projects/hypnomed/data/template/fsaverage/surf/lh.inflated'],['/home/romy.beaute/projects/hypnomed/data/template/fsaverage/surf/rh.inflated']});
d.surf.pial = SurfStatReadSurf({['/home/romy.beaute/projects/hypnomed/data/template/fsaverage/surf/lh.pial'],['/home/romy.beaute/projects/hypnomed/data/template/fsaverage/surf/rh.pial']});

d.cortex = load('/home/romy.beaute/projects/hypnomed/data/cortex.mat');
d.cortex = squeeze(d.cortex.cortex') + 1;


%cf create_csv_files.m to see how they were created
% d.expertise = dlmread('./expertise_covs.csv',',',1,0);
d.state =dlmread('/home/romy.beaute/projects/hypnomed/analysis/scripts_stats/state_covs.csv',',',1,0);
d.subject =dlmread('/home/romy.beaute/projects/hypnomed/analysis/scripts_stats/subject_covs.csv',',',1,0);

for outlier_index = 1:length(p.outliers)
    outlier_index_in_csv = d.subject==p.outliers(outlier_index);
    d.state(outlier_index_in_csv,:) = [];
    % d.expertise(outlier_index_in_csv,:) = [];
    d.subject(outlier_index_in_csv,:) = [];
    d.emb(outlier_index_in_csv,:,:) = [];
end

% d.expertise = term(d.expertise);
d.state = term(d.state);
d.subject = term(var2fac(d.subject));


% emb_states = cell(2,1);
% state_names = cell(2,1);
% mean_emb_states = cell(2,1);
% for group_id = 1:2
%     [emb_states{group_id},state_names{group_id}] = get_state_group(d,p.states_wanted{group_id},p.groups_wanted{group_id},p.interaction); %states_wanted = 'COMP','OP','RS'  // group_wanted = 'experts','novices','all'
% end
% if ~strcmp(p.groups_wanted{1},p.groups_wanted{2})
%     state_names{1} = p.groups_wanted{1};
%     state_names{2} = p.groups_wanted{2};
% end
% [contrast,contrastsList] = get_contrast(d);

% disp(contrastsList{contrast_id})




emb_states = cell(2,1);
state_names = cell(2,1);
mean_emb_states = cell(2,1);


for group_id = 1:2
    [emb_states{group_id},state_names{group_id}] = get_state_group(d,p.states_wanted{group_id},p.interaction); %states_wanted = 'COMP','OP','RS'  // group_wanted = 'experts','novices','all'
end
% if ~strcmp(p.groups_wanted{1},p.groups_wanted{2})
%     state_names{1} = p.groups_wanted{1};
%     state_names{2} = p.groups_wanted{2};
% end
[contrast,contrastsList] = get_contrast(d);

disp(contrastsList{contrast_id})






% tmp_reports = cell(length(p.dims_wanted),1);

%%see where are the peaks of clusters (MNI coord [x,y,z])
% tableau avec peak des clusters : take coordonnées MNI et sort le nom de la région
% nb de voxels du cluster
for dim_id = 1:length(p.dims_wanted)
    dim = p.dims_wanted(dim_id);
    [stats_p,stats_n,maskRoi] = get_mixed_effect(d,p,dim,contrast(:,contrast_id));
    %display coord of significant clusters:
    for stats = [stats_p,stats_n]
        clusid = stats.peak.clusid;
        significant_clusid = stats.clus.clusid(stats.clus.P<0.05);
        for cluster_id = 1:length(significant_clusid)
            peak.t =  max(stats.peak.t(ismember(clusid,cluster_id)));
            peak.vertid = stats.peak.vertid(stats.peak.t==peak.t);
            peak.clusid = cluster_id;
            peak.P = stats.clus.P(cluster_id);
            peak.kL = stats.clus.nverts(cluster_id); %nb of voxels
            term( peak ) + term( SurfStatInd2Coord( peak.vertid, d.surf.pial )', {'x','y','z'})
            test_inTPJ = (stats.pval.C<=0.05)' + TPJ_mask;

            
        end
        
        disp('Nb of voxels in TPJ mask:')
        disp(sum(test_inTPJ == 2))

    end
    
end




function [emb_state,state_name] = get_state_group(d,state_wanted,interaction)

    if interaction>0
        for state_id = 1:3
            d.emb(logical(d.state(state_id)),:,:) = d.emb(logical(d.state(state_id)),:,:) - d.emb(logical(d.state(interaction)),:,:);
        end
    end
    
    if strcmp(state_wanted,'control')
        state_name = "control";
        state_index = d.state(1);
    elseif strcmp(state_wanted,'meditation')
        state_name = "meditation";
        state_index = d.state(2);
    elseif strcmp(state_wanted,'hypnose')
        state_name = "hypnose";
        state_index = d.state(3);
    % elseif strcmp(state_wanted,'MED')
    %     state_name = "MED STATES";
    %     state_index = d.state(1) | d.state(2);
    % elseif strcmp(state_wanted,'OP+RS')
    %     state_name = "OPEN PRESENCE + RESTING STATE";
    %     state_index = d.state(2) | d.state(3);
    end
    
    % if strcmp(group_wanted,'experts')
    %     emb_state = zeros(sum(d.expertise(2)&state_index),20484,size(d.emb,3)); %define the first d.state which will be compared with the second d.state
    %     state_index = state_index & d.expertise(2);
    % elseif strcmp(group_wanted,'novices')
    %     emb_state = zeros(sum(d.expertise(1)&state_index),20484,size(d.emb,3)); %define the first d.state which will be compared with the second d.state
    %     state_index = state_index & d.expertise(1);
    % elseif strcmp(group_wanted,'all')
    %     emb_state = zeros(sum(state_index),20484,size(d.emb,3)); %define the first d.state which will be compared with the second d.state
    %     state_index = logical(state_index);
    % end

    % if strcmp(group_wanted,'experts')
    %     emb_state = zeros(sum(d.expertise(2)&state_index),20484,size(d.emb,3)); %define the first d.state which will be compared with the second d.state
    %     state_index = state_index & d.expertise(2);
    % elseif strcmp(group_wanted,'novices')
    %     emb_state = zeros(sum(d.expertise(1)&state_index),20484,size(d.emb,3)); %define the first d.state which will be compared with the second d.state
    %     state_index = state_index & d.expertise(1);
    % elseif strcmp(group_wanted,'all')
    emb_state = zeros(sum(state_index),20484,size(d.emb,3)); %define the first d.state which will be compared with the second d.state
    state_index = logical(state_index);
    % end
    
    emb_state(:,d.cortex,:) = d.emb(logical(state_index),:,:);
end




function [contrast,contrastsList] = get_contrast(d)
    
    contrast=zeros(size(d.emb,1),3);

    %Control vs Meditation
    contrast(:,1) = d.state(1)-d.state(2);

    %Control vs Hypnose
    contrast(:,2) = d.state(1)-d.state(3);

    %Meditation vs Hypnose
    contrast(:,3) = d.state(2)-d.state(3);

    contrastsList = {'control_vs_meditation', 'control_vs_hypnose', 'meditation_vs_hypnose'};

end;



function [stats_p,stats_n,maskRoi] = get_mixed_effect(d,p,dim_id,used_contrast)
    mask = zeros(1,size(d.emb,2));
    mask(d.cortex) = 1;
    negative_contrast = used_contrast * -1;
%add the identity matrix I to allow for independent "white" noise in every observation (this is added by default to any fixed effect model, but it must be specifically added to a mixed effects model)
    % M = 1 + d.expertise + d.state + d.expertise*d.state+ random(d.subject) + I;
    % M = 1 + d.state + d.expertise*d.state+ random(d.subject) + I;
    M = 1 + d.state + random(d.subject) + I;
    surface_emb = rand(size(d.emb,1), 20484);
    surface_emb(:,d.cortex) = squeeze(d.emb(:,:,dim_id));
    slm  = SurfStatLinMod(surface_emb, M, d.surf.pial);
    %get P-values for positive contrast corrected for multiple comparisons (RFT)

    slmp  = SurfStatT(slm, used_contrast);
    stats_p = struct();
    [stats_p.pval,stats_p.peak,stats_p.clus,stats_p.clusid] = SurfStatP(slmp, mask, p.clusp);

    %Get significant clusters id to use them as mask later
    significant_clusters_ids_p = stats_p.clus.clusid(stats_p.clus.P<0.05);
    maskRoipos = zeros(1,20484);

    for significant_clusters_index = 1:length(significant_clusters_ids_p)
        maskRoipos = maskRoipos + (stats_p.clusid == significant_clusters_ids_p(significant_clusters_index)); 
    end 
    maskRoipos = logical(maskRoipos);

    %get P-values for negative contrast corrected for multiple comparisons (RFT)

    slmn  = SurfStatT(slm, negative_contrast);
    stats_n = struct();
    [stats_n.pval,stats_n.peak,stats_n.clus,stats_n.clusid] = SurfStatP(slmn, mask, p.clusp);

    %Get significant clusters id to use them as mask later
    significant_clusters_ids_n = stats_n.clus.clusid(stats_n.clus.P<0.05);
    maskRoineg = zeros(1,20484);

    for significant_clusters_index = 1:length(significant_clusters_ids_n)
         maskRoineg = maskRoineg + (stats_n.clusid == significant_clusters_ids_n(significant_clusters_index)); 
    end
    maskRoineg = logical(maskRoineg);

    %Fused mask of pos and neg contrast
    maskRoi = logical(maskRoineg+maskRoipos); 
end




% function [tmp_file_name] = make_figs(p,d,dim_id,emb_states,state_names,stats_p,stats_n,maskRoi)

%     report_description = {sprintf('%s analysis framework',p.analysis_framework),'',sprintf('Comparison between %s and %s (%s only)',state_names{1},state_names{2},p.prefixe)};

%     tmp_file_name = sprintf('figures/tmp_report_dim%0d', dim_id);
%     mean_state1_dim = emb_states{1};
%     mean_state1_dim = mean(squeeze(mean_state1_dim(:,:,dim_id)),1);
%     mean_state2_dim = emb_states{2};
%     mean_state2_dim = mean(squeeze(mean_state2_dim(:,:,dim_id)),1);

%     figure('Position',[20 20 1280 720])
%     text(0.5,0.5,report_description,'VerticalAlignment','middle','HorizontalAlignment','center','FontSize', 28);
%     set(gca,'visible','off')
%     export_fig(tmp_file_name,'-pdf') 
%     close all

%     figure('Position',[20 20 1280 720])
%     text(0.5,0.5,sprintf('Dimension %0d',dim_id),'VerticalAlignment','middle','HorizontalAlignment','center','FontSize', 28);
%     set(gca,'visible','off')
%     export_fig(tmp_file_name,'-pdf','-append')  
%     close all

%     %Mean gradient values for the compassion d.state
%     figure('Position', [20 20 1280 720]); 
%     axes = SurfStatView(mean_state1_dim, d.surf.pial,{'Mean gradient values'});
%     SurfStatColormap('jet');
%     SurfStatColLim(p.col_lim_dim);
%     ttl=title(sprintf('%s Gradient (Dim%0d)',state_names{1},dim_id),'FontSize', 14);
%     ttl.Parent = axes(2);
%     set(ttl,'position',get(ttl,'position')+[0 87 0]);
%     export_fig(tmp_file_name,'-pdf','-append')  
%     close all


%     %Mean gradient values for the resting d.state
%     figure('Position', [20 20 1280 720]); 
%     axes = SurfStatView(mean_state2_dim, d.surf.pial,{'Mean gradient values'});
%     SurfStatColormap('jet');
%     SurfStatColLim(p.col_lim_dim);
%     ttl=title(sprintf('%s Gradient (Dim%0d)',state_names{2},dim_id),'FontSize', 14);
%     ttl.Parent = axes(2);
%     set(ttl,'position',get(ttl,'position')+[0 87 0]);
%     export_fig(tmp_file_name,'-pdf','-append')  
%     close all


%     %positive contrast
%     figure('Position', [20 20 1280 720]);
%     axes = SurfStatView(stats_p.pval, d.surf.pial,{'Significant clusters', sprintf('Contrast: %s>%s',state_names{1},state_names{2})});
%     ttl=title(sprintf('Significant clusters and vertices for dimension %0d (%s>%s)  (corrected with RFT)',dim_id,state_names{1},state_names{2}),'FontSize', 14);
%     ttl.Parent = axes(2);
%     set(ttl,'position',get(ttl,'position')+[0 87 0]);
%     export_fig(tmp_file_name,'-pdf','-append')  
%     close all


%     %negative contrast 
%     figure('Position', [20 20 1280 720]);
%     axes = SurfStatView(stats_n.pval, d.surf.pial,{'Significant clusters', sprintf('Contrast: %s<%s',state_names{1},state_names{2})});
%     ttl=title(sprintf('Significant clusters and vertices for dimension %0d (%s<%s)  (corrected with RFT)',dim_id,state_names{1},state_names{2}),'FontSize', 14);
%     ttl.Parent = axes(2);
%     set(ttl,'position',get(ttl,'position')+[0 87 0]);
%     export_fig(tmp_file_name,'-pdf','-append')  
%     close all


%     %plot mean gradient values for the d.state 1 inside all significant
%     %clusters
%     figure('Position', [20 20 1280 720]); 
%     axes = SurfStatView(mean_state1_dim.*maskRoi, d.surf.pial,{sprintf('Mean gradient values (Dim%0d)',dim_id)});
%     SurfStatColormap('jet');
%     SurfStatColLim(p.col_lim_dim);
%     ttl=title(sprintf('%s gradient (significant clusters only)',state_names{1}),'FontSize', 14);
%     ttl.Parent = axes(2);
%     set(ttl,'position',get(ttl,'position')+[0 87 0]);
%     export_fig(tmp_file_name,'-pdf','-append')  
%     close all


%     %%plot Difference between mean gradient values for the resting d.state of the Compassion and resting states inside all significant
%     %clusters
%     figure('Position', [20 20 1280 720]); 
%     axes = SurfStatView((mean_state1_dim-mean_state2_dim).*maskRoi, d.surf.pial,{sprintf('%s - %s mean gradient values (Dim%0d)',state_names{1},state_names{2},dim_id)});
%     SurfStatColormap('jet');
%     SurfStatColLim(p.diff_lim);
%     ttl=title(sprintf('Difference of gradient between %s and %s (significant clusters only)',state_names{1},state_names{2}),'FontSize', 14);
%     ttl.Parent = axes(2);
%     set(ttl,'position',get(ttl,'position')+[0 87 0]);
%     export_fig(tmp_file_name,'-pdf','-append')  
%     close all

%     %%plot Difference between mean gradient values for the resting d.state of the
%     %%Compassion and resting states across all voxels
%     figure('Position', [20 20 1280 720]); 
%     axes = SurfStatView((mean_state1_dim-mean_state2_dim), d.surf.pial,{sprintf('%s - %s mean gradient values (Dim%0d)',state_names{1},state_names{2}, dim_id)});
%     SurfStatColormap('jet');
%     SurfStatColLim(p.diff_lim);
%     ttl=title(sprintf('Difference of gradient between %s and %s',state_names{1},state_names{2}),'FontSize', 14);
%     ttl.Parent = axes(2);
%     set(ttl,'position',get(ttl,'position')+[0 87 0]);
%     export_fig(tmp_file_name,'-pdf','-append')  
%     close all


%     bins_nb = 50;
%     figure('Position', [20 20 1280 720]);
%     h1 = histfit(mean_state1_dim(d.cortex)',bins_nb,'kernel');
%     h1(2).Color = [.2 .2 .9];
%     hold on
%     h2 = histfit(mean_state2_dim(d.cortex)',bins_nb,'kernel');
%     h2(2).Color = [.9 .2 .2];
%     xlabel(sprintf('Dimension %0d Mean Gradient Score',dim_id),'FontSize', 18)
%     ylabel('\fontsize{18}Frequency')
%     legend([h1(2),h2(2)],{state_names{1}, state_names{2}}, 'Fontsize',16)
%     title(sprintf('Histogram of the %s and %s gradients (Dim%0d)',state_names{1},state_names{2},dim_id),'FontSize', 20);
%     hold off
%     alpha(0.4)
%     export_fig(tmp_file_name,'-pdf','-append')  
%     close all

% end
