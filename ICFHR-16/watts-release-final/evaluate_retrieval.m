function mAP = evaluate_retrieval(opts,data,embedding)

mAP.cca = [];
mAP.kcca = [];
mAP.hybrid = [];

if opts.TestCCA
    mAP.cca = evaluateCCA(opts, data, embedding.cca);
end

if opts.TestKCCA
    mAP.kcca = evaluateKCCA(opts, data, embedding.kcca);
end

if opts.TestHybrid
    mAP.hybrid = evaluateHybrid(opts, data, embedding.cca);
end

end
