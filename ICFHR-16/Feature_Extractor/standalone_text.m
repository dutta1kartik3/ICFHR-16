function [final_vec_text] = standalone_text(text,dataset)
   %text should be a cell array of words (say there are 'n' words), otherwise the code breaks
   %final_vec_text is 96xn, for GW, 200xn for others

    if(dataset == 1)
        load embedding-iam
        load attModels-iam
    end
    
    if(dataset == 2)
        load embedding-gw
        load attModels-gw
    end
        
    if(dataset == 3)
        load embedding-bentham
        load attModels-bentham
    
    end

    %% Please do not change these paramaters, otherwise the whole code fails
    levels = [2, 3, 4, 5, 6, 7, 8, 9, 10];
    levelsB = [2, 3, 4, 5, 6];
    if(dataset == 2)
        levels = [2 3 4 5];
        levelsB = [2];
    end
    numBigrams = 50;
    fid = fopen('bigrams.txt','r');
    bgrams = textscan(fid,'%s');
    fclose(fid);
    bgrams = bgrams{1}(1:numBigrams);
    voc = ['abcdefghijklmnopqrstuvwxyz' '0123456789'];
    str2cell = @(x) {char(x)};
    voc = arrayfun(str2cell, voc);
    %%
    
    %convert the char array to cell, if not already
    %text = cellstr(text);
    
    
    %Get phoc rep. of unigrams, bigrams and then concat.
    phocuni = phoc_mex(text, voc, int32(levels));
    phocbi = phoc_mex(text, bgrams, int32(levelsB));
    phoc = [phocuni;phocbi];
    
    %L2 norm
    phoc = bsxfun(@rdivide, phoc,sqrt(sum(phoc.*phoc)));
    
    %Mean subtraction
    final_vec_text = bsxfun(@minus, phoc,embedding.mphocs);
    
    %Embedding into common subspace
    final_vec_text = embedding.Wy(:,1:embedding.K)' * final_vec_text;
    %L2 norm
    final_vec_text = bsxfun(@rdivide, final_vec_text,sqrt(sum(final_vec_text.*final_vec_text)));
    
end
