condition = 'run-3'

emb_states= sprintf('/home/romy.beaute/projects/hypnomed/diffusion_embedding/emb_matrices/group/group_%s_embedding.mat',condition)

% d.subject =dlmread('/home/romy.beaute/projects/hypnomed/analysis/scripts_stats/data/subject_covs.csv',',',1,0);



n_scans = 39;

emb = load(emb_states)


subs_emb = emb.subs;
emb = emb.emb;

mean_emb= mean(emb,1);

corr_emb = zeros(2,2,n_scans);
for i=1:n_scans
[R, P]= corrcoef(squeeze(mean_emb(:,:,1)), squeeze(emb(i,:,1)));
corr_emb(:,:,i)= R;
end


%plot of correlation coefficients
X = linspace(0,1.5,1000);

cf = figure
corr_plot= histfit(squeeze(corr_emb(1,2,:)));
saveas(cf,sprintf('/home/romy.beaute/projects/hypnomed/analysis/results/fisher_outliers/corrplot_fisher_%s.png',condition))

%plot of fisher z-transformed correlation coefficients

Z = zeros(1,n_scans);
for i=1:n_scans
r = corr_emb(1,2,i);
Z(:,i) = .5*log10(r+1)/log10(exp(1))-.5*log10(1-r)/log10(exp(1));
%could have used atanh() function which computes the same
end

mean_z = mean(Z,2);
std_z = std(Z,0,2);
under_lim = mean_z - 2.5*std_z;
super_lim = mean_z + 2.5*std_z;
X = linspace(0,2,1000);

h = figure
density = histfit(Z , 50); % 'Normalization', 'Probability');
line([mean_z mean_z], [0 n_scans], 'color','r');
line([under_lim under_lim], [0 n_scans], 'color','r');
line([super_lim super_lim], [0 n_scans], 'color','r');
saveas(h,sprintf('/home/romy.beaute/projects/hypnomed/analysis/results/fisher_outliers/density_outliers_fisher_%s.png',condition))
n_scans = lentgh(subjectsList)

outliers = zeros(n_scans,1);
for i=1:n_scans
    if Z(:,i)< under_lim
        outliers(i,:) = 1;
    else
        outliers(i,:) = 0;
    end
end

subjectsList = {
    'sub-01'
    'sub-02'
    'sub-03'
    'sub-04'
    'sub-05'
    'sub-06'
    'sub-07'
    'sub-08'
    'sub-09'
    'sub-10'
    'sub-11'
    'sub-12'
    'sub-13'
    'sub-14'
    'sub-15'
    'sub-16'
    'sub-17'
    'sub-18'
    'sub-19'
    'sub-20'
    'sub-21'
    'sub-22'
    'sub-23'
    'sub-24'
    'sub-25'
    'sub-26'
    'sub-27'
    'sub-28'
    'sub-29'
    'sub-30'
    'sub-31'
    'sub-32'
    'sub-33'
    'sub-34'
    'sub-35'
    'sub-36'
    'sub-37'
    'sub-38'
    'sub-39'
    'sub-40'};

if strcmp(condition,'run-3')
    subjectsList = {
    'sub-01'
    'sub-02'
    'sub-03'
    'sub-04'
    'sub-05'
    'sub-06'
    'sub-07'
    'sub-08'
    'sub-09'
    'sub-10'
    'sub-11'
    'sub-12'
    'sub-13'
    'sub-14'
    'sub-15'
    'sub-16'
    'sub-17'
    'sub-18'
    'sub-19'
    'sub-20'
    'sub-21'
    'sub-22'
    'sub-23'
    'sub-24'
    'sub-25'
    'sub-26'
    'sub-27'
    'sub-28'
    'sub-29'
    'sub-30'
    'sub-31'
    'sub-33'
    'sub-34'
    'sub-35'
    'sub-36'
    'sub-37'
    'sub-38'
    'sub-39'
    'sub-40'};
end


disp(subjectsList(logical(outliers)))
sub_out = outliers;
outliers =num2cell(outliers);

out_tab = cat(2, subjectsList, outliers);

filename = sprintf('/home/romy.beaute/projects/hypnomed/analysis/results/fisher_outliers/outliers_fisher_%s.csv',condition);    %must end in csv
writetable( cell2table(out_tab), filename, 'writevariablenames', false, 'quotestrings', true)


disp(subjectsList(logical(sub_out)))