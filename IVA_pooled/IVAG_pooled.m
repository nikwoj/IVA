% Apply IVA-G to the subjects in a given site, save resulting W matrix

[subjs] = textread('seed_subjs.txt');


X = load(sprintf('SCV_IVA_pcawhitened_subj%d.mat', subjs));
X = X.X_white;

W = zeros(20,20,subjs);
for kk=1:subjs
    W(:,:,kk) = eye(20);
end

W = icatb_iva_second_order(X,'verbose',true,'whiten',false);
save(sprintf('W_IVA_G_si%d_su%d_site%d.mat',1,subjs,1), 'W');

exit();
