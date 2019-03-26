function [final_vec_img] = standalone_img(imagesBatch,dataset)
    %img should be a cell array of grayscale images (say there are 'n' images), otherwise the code breaks
    %final_vec_img is 96xn for GW, 200xn for others

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
    %Last row contains bias values of SVM, not required
    attModels = attModels(1:2048,:);
    
    %Getting HWNet Finetuned features
    if(dataset == 1)
        cnn = cnnJaderbergRepwatts(imagesBatch,'IAM.ini')'; %Aug one for better performance !! ?
    end
    
    if(dataset == 2)
        cnn = cnnJaderbergRepwatts(imagesBatch,'GW.ini')';
    end
        
    if(dataset == 3)
        cnn = cnnJaderbergAug(imagesBatch,'Bentham.ini')';    
    end
    
    %Getting the embedded attribute features
    attRepr = attModels'*cnn;

    % L2 normalization
    attRepr = bsxfun(@rdivide, attRepr,sqrt(sum(attRepr.*attRepr)));
    attRepr(isnan(attRepr)) = 0;
    
    %Subtraction with mean vector
    attRepr =  bsxfun(@minus, attRepr,embedding.matts);

    % Embedding into common subspace
    attRepr = embedding.Wx(:,1:embedding.K)' * attRepr;
    
    %L2 normalization
    final_vec_img = (bsxfun(@rdivide, attRepr, sqrt(sum(attRepr.*attRepr))));

end
