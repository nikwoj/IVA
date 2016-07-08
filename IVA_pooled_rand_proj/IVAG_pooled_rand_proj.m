% Apply IVA-G to the subjects in a given site, save resulting W matrix

%[subjs] = textread('seed_subjs.txt');

[subjs, seed] = textread('seed_subjs.txt');

A = load(sprintf('SCV_IVA_rand_proj_W_subj%d_start%d.mat', subjs, seed));
X = A.X;
W = A.W;

W = icatb_iva_second_order(X,'verbose',true,'whiten',false,'W_init',W);
save(sprintf('W_IVAG_pooled_rand_proj_subj%d_start%d.mat',subjs,seed), 'W');


exit();
