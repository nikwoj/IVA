% Apply IVA-G to the subjects in a given site, save resulting W matrix

%[subjs] = textread('seed_subjs.txt');

[subjs, seed] = textread('seed_subjs.txt');

A = load(sprintf('SCV_IVA_pcawhitened_W_seed%d_subj%d.mat', seed, subjs));
X = A.X_white;
W = A.W;

%X = load(sprintf('SCV_IVA_pcawhitened_subj%d.mat', subjs));
%X = X.X_white;

%W = zeros(20,20,subjs);
%for kk=1:subjs
%    W(:,:,kk) = eye(20);
%end

W = icatb_iva_second_order(X,'verbose',true,'whiten',false,'W_init',W);
save(sprintf('W_IVAG_pooled_su%d_start%d.mat',subjs,seed), 'W');


exit();
