function hybrid_map = evaluateHybrid(opts,DATA,embedding)

fprintf('\n');
disp('**************************************');
disp('************  Hybrid CSR  ***********');
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
alpha = 0:0.1:1;
hybrid_map = zeros(length(alpha),1);
hybrid_p1 = zeros(length(alpha),1);
for i=1:length(alpha)
    attRepr_hybrid = attReprTe_cca*alpha(i) + phocsTe_cca*(1-alpha(i));
    [p1,mAPEucl,q] = eval_dp_asymm(opts,attRepr_hybrid,attReprTe_cca,DATA.wordClsTe,DATA.labelsTe);
    hybrid_map(i) = mean(mAPEucl)*100;
    hybrid_p1(i) = mean(p1)*100;
end

[best_map,idx] = max(hybrid_map);
best_p1 = hybrid_p1(idx);
best_alpha = alpha(idx);

% Display info
disp('------------------------------------');
fprintf('alpha: %.2f reg: %.8f. k: %d\n', best_alpha, embedding.reg, embedding.K);
fprintf('hybrid --   test: (map: %.2f. p@1: %.2f)\n',  best_map, best_p1);
disp('------------------------------------');

plot(alpha,hybrid_map,'.-','MarkerSize',16)
title(opts.dataset)
xlabel('alpha')
ylabel('Mean Average Precision (%)')
grid on

end
