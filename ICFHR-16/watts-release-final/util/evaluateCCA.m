function mAP = evaluateCCA(opts,DATA,embedding)
% Evaluate CCA.

fprintf('\n');
disp('**************************************');
disp('***************  CSR   ***************');
disp('**************************************');
opts.test = 1;

% A) L2 normalize and mean center. Not critical, but helps a bit.
attReprTe = bsxfun(@rdivide, DATA.attReprTe,sqrt(sum(DATA.attReprTe.*DATA.attReprTe)));
attReprTe(isnan(attReprTe)) = 0;
phocsTe = bsxfun(@rdivide, DATA.phocsTe,sqrt(sum(DATA.phocsTe.*DATA.phocsTe)));

attReprTe =  bsxfun(@minus, attReprTe,embedding.matts);
phocsTe=  bsxfun(@minus, phocsTe,embedding.mphocs);

% Embed  test
attReprTe_cca = embedding.Wx(:,1:embedding.K)' * attReprTe;
phocsTe_cca = embedding.Wy(:,1:embedding.K)' * phocsTe;

% L2 normalize (critical)
attReprTe_cca = (bsxfun(@rdivide, attReprTe_cca, sqrt(sum(attReprTe_cca.*attReprTe_cca))));
phocsTe_cca = (bsxfun(@rdivide, phocsTe_cca, sqrt(sum(phocsTe_cca.*phocsTe_cca))));

% Evaluate
% QBE
[p1,mAPEucl, q] = eval_dp_asymm(opts,attReprTe_cca, attReprTe_cca,DATA.wordClsTe,DATA.labelsTe);
qbe_test_map = mean(mAPEucl);
qbe_test_p1 = mean(p1);

% QBS (note the 1 at the end)
[p1,mAPEucl, q] = eval_dp_asymm(opts,phocsTe_cca, attReprTe_cca,DATA.wordClsTe,DATA.labelsTe,1);
qbs_test_map = mean(mAPEucl);
qbs_test_p1 = mean(p1);

% Display info
disp('------------------------------------');
fprintf('reg: %.8f. k: %d\n',  embedding.reg, embedding.K);
fprintf('qbe --   test: (map: %.2f. p@1: %.2f)\n',  100*qbe_test_map, 100*qbe_test_p1);
fprintf('qbs --   test: (map: %.2f. p@1: %.2f)\n',  100*qbs_test_map, 100*qbs_test_p1);
disp('------------------------------------');

mAP.qbe = 100*qbe_test_map;
mAP.qbs = 100*qbs_test_map;

if(strcmp(opts.dataset,'IAM')==1)

	% Invocab	

	opts.invocab = 1;
	[p1,mAPEucl, q] = eval_dp_asymm(opts,attReprTe_cca,attReprTe_cca,DATA.wordClsTe,DATA.labelsTe);
	qbe_test_map = mean(mAPEucl);
	qbe_test_p1 = mean(p1);

	% QBS (note the 1 at the end)
	[p1,mAPEucl, q] = eval_dp_asymm(opts,phocsTe_cca,attReprTe_cca,DATA.wordClsTe,DATA.labelsTe,1);
	qbs_test_map = mean(mAPEucl);
	qbs_test_p1 = mean(p1);

	% Display info

	disp('------------------------------------');
	disp('In Vocabulary');
	fprintf('reg: %.8f. k: %d\n',  embedding.reg, embedding.K);
	fprintf('qbe --   test: (map: %.2f. p@1: %.2f)\n',  100*qbe_test_map, 100*qbe_test_p1);
	fprintf('qbs --   test: (map: %.2f. p@1: %.2f)\n',  100*qbs_test_map, 100*qbs_test_p1);
	disp('------------------------------------');

	opts.invocab = 0;

	%% OOV

	opts.outvocab = 1;

	[p1,mAPEucl, q] = eval_dp_asymm(opts,attReprTe_cca,attReprTe_cca,DATA.wordClsTe,DATA.labelsTe);
	qbe_test_map = mean(mAPEucl);
	qbe_test_p1 = mean(p1);

	% QBS (note the 1 at the end)
	[p1,mAPEucl, q] = eval_dp_asymm(opts,phocsTe_cca,attReprTe_cca,DATA.wordClsTe,DATA.labelsTe,1);
	qbs_test_map = mean(mAPEucl);
	qbs_test_p1 = mean(p1);

	% Display info

	disp('------------------------------------');
	disp('Out of Vocabulary');
	fprintf('reg: %.8f. k: %d\n',  embedding.reg, embedding.K);
	fprintf('qbe --   test: (map: %.2f. p@1: %.2f)\n',  100*qbe_test_map, 100*qbe_test_p1);
	fprintf('qbs --   test: (map: %.2f. p@1: %.2f)\n',  100*qbs_test_map, 100*qbs_test_p1);
	disp('------------------------------------');

	opts.outvocab = 0;
end

end
