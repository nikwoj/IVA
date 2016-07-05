% Apply IVA-G to the subjects in a given site, save resulting W matrix

[subj_site, num_sites] = textread('num_subj_site.txt');
%subj_site = 2;
%num_sites = 2

X = zeros(20, 32968, subj_site);
files = dir('*pcawhitened*.mat');

files = {files.name};

W = zeros(20,20,subj_site);
for kk=1:subj_site
    W(:,:,kk) = eye(20);
end

for k=1:num_sites
    for kk=1:subj_site
        b = load(files{kk});
        X(:,:,kk) = b.X_white;
    end
    W = icatb_iva_second_order(X,'verbose',true,'opt_approach','quasi');
    save(sprintf('W_IVA_G_si%d_su%d_site%d.mat',num_sites,subj_site,k), 'W');
end

exit();
