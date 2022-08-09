import numpy as np
emb = scipy.io.loadmat('/dycog/meditation/ERC/Analyses/MINCOM/diffusion_embedding_step/matrix_data/co_om_rs_group_embedding_minus_50.mat')
subs_emb = emb.subs
emb = emb.emb
mean_emb = mean(emb,1)
corr_emb = np.zeros((2,2,225))
for i in np.arange(1,225+1).reshape(-1):
    R,P = corrcoef(np.squeeze(mean_emb(:,:,1)),np.squeeze(emb(i,:,1)))
    corr_emb[:,:,i] = R

#plot of correlation coefficients
X = np.linspace(0,1.5,1000)
figure
corr_plot = histfit(np.squeeze(corr_emb(1,2,:)))
#plot of fisher z-transformed correlation coefficients

Z = np.zeros((1,225))
for i in np.arange(1,225+1).reshape(-1):
    r = corr_emb(1,2,i)
    Z[:,i] = 0.5 * log10(r + 1) / log10(np.exp(1)) - 0.5 * log10(1 - r) / log10(np.exp(1))
    #could have used atanh() function which computes the same

mean_z = mean(Z,2)
std_z = std(Z,0,2)
under_lim = mean_z - 1.6 * std_z
super_lim = mean_z + 1.6 * std_z
X = np.linspace(0,2,1000)
figure
density = histfit(Z,50)

line(np.array([mean_z,mean_z]),np.array([0,40]),'color','r')
line(np.array([under_lim,under_lim]),np.array([0,40]),'color','r')
line(np.array([super_lim,super_lim]),np.array([0,40]),'color','r')
outliers = np.zeros((225,1))
for i in np.arange(1,225+1).reshape(-1):
    if Z(:,i) < under_lim:
        outliers[i,:] = 1
    else:
        outliers[i,:] = 0

subjectsList = np.array(['sub-002','sub-002','sub-002','sub-004','sub-004','sub-004','sub-005','sub-005','sub-005','sub-007','sub-007','sub-007','sub-010','sub-010','sub-010','sub-011','sub-011','sub-011','sub-012','sub-012','sub-012','sub-014','sub-014','sub-014','sub-016','sub-016','sub-016','sub-017','sub-017','sub-017','sub-018','sub-018','sub-018','sub-022','sub-022','sub-022','sub-025','sub-025','sub-025','sub-026','sub-026','sub-026','sub-028','sub-028','sub-028','sub-029','sub-029','sub-029','sub-030','sub-030','sub-030','sub-032','sub-032','sub-032','sub-034','sub-034','sub-034','sub-035','sub-035','sub-035','sub-036','sub-036','sub-036','sub-037','sub-037','sub-037','sub-038','sub-038','sub-038','sub-040','sub-040','sub-040','sub-042','sub-042','sub-042','sub-052','sub-052','sub-052','sub-053','sub-053','sub-053','sub-054','sub-054','sub-054','sub-055','sub-055','sub-055','sub-056','sub-056','sub-056','sub-057','sub-057','sub-057','sub-058','sub-058','sub-058','sub-059','sub-059','sub-059','sub-060','sub-060','sub-060','sub-062','sub-062','sub-062','sub-063','sub-063','sub-063','sub-064','sub-064','sub-064','sub-065','sub-065','sub-065','sub-067','sub-067','sub-067','sub-068','sub-068','sub-068','sub-069','sub-069','sub-069','sub-070','sub-070','sub-070','sub-071','sub-071','sub-071','sub-072','sub-072','sub-072','sub-073','sub-073','sub-073','sub-074','sub-074','sub-074','sub-075','sub-075','sub-075','sub-076','sub-076','sub-076','sub-077','sub-077','sub-077','sub-078','sub-078','sub-078','sub-079','sub-079','sub-079','sub-080','sub-080','sub-080','sub-081','sub-081','sub-081','sub-082','sub-082','sub-082','sub-083','sub-083','sub-083','sub-087','sub-087','sub-087','sub-089','sub-089','sub-089','sub-090','sub-090','sub-090','sub-091','sub-091','sub-091','sub-092','sub-092','sub-092','sub-093','sub-093','sub-093','sub-094','sub-094','sub-094','sub-095','sub-095','sub-095','sub-096','sub-096','sub-096','sub-097','sub-097','sub-097','sub-098','sub-098','sub-098','sub-099','sub-099','sub-099','sub-101','sub-101','sub-101','sub-102','sub-102','sub-102','sub-103','sub-103','sub-103','sub-104','sub-104','sub-104','sub-105','sub-105','sub-105','sub-106','sub-106','sub-106','sub-108','sub-108','sub-108','sub-109','sub-109','sub-109'])
print(subjectsList(logical(outliers)))
outliers = num2cell(outliers)
out_tab = cat(2,subjectsList,outliers)