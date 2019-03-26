function extract_FV_features_fast(opts)

imagesTOC = readImagesToc(opts.fileImages);
nWords = length(imagesTOC);

imagesPerBatch = nWords;
nBatches = int32(ceil(nWords/imagesPerBatch));
featsBatch = zeros(opts.FVdim,imagesPerBatch,'single');

% Write output header
fid = fopen(opts.fileFeatures, 'w');
fwrite(fid, nWords, 'int32');
fwrite(fid, opts.FVdim, 'int32');
fclose(fid);
tic;
for cb=1:nBatches    
    sp = (cb-1)*imagesPerBatch + 1;
    ep = sp + imagesPerBatch -1;
    if ep > nWords
        ep = nWords;
    end
    nInBatch = ep-sp+1;
    fprintf('Extracting FV batch %d/%d (%d images)\n',cb,nBatches,nInBatch);        
    % Read image batch
    
    [fid,msg] = fopen(opts.fileImages, 'r');
    readIm = @(x) readImage(fid, imagesTOC, x);
    imagesBatch = arrayfun(readIm, [sp:ep], 'uniformOutput', false);
    for i=1:size(imagesBatch,2)
	imagesBatch{i} = im2single(imagesBatch{i});
    end
    
    if(strcmp(opts.dataset,'GW')==1)
    	temp =  cnnJaderbergRepwatts(imagesBatch,'GW.ini')';
	
    elseif(strcmp(opts.dataset, 'IAM')==1)
	%temp = cnnJaderbergAug(imagesBatch,'IAM.ini')';
	load /lustre/kartik/iam-final.mat
	temp = augFeat';
    else
	temp = cnnJaderbergAug(imagesBatch,'Bentham.ini')';
    end
    
    fclose(fid);
	    
    parfor i=1:length(imagesBatch)        
        featsBatch(:,i) = single(temp(:,i));
    end
    featsBatch(isnan(featsBatch)) = 0;
     
    % Write the batch
    fid = fopen(opts.fileFeatures, 'r+');    
    fseek(fid, 2*4  + (int64(cb)-1)*imagesPerBatch*opts.FVdim * 4, 'bof');
    fwrite(fid, featsBatch(:,1:nInBatch), 'single');
end
disp(toc);
end

% -------------------------------------------------------------------------
function fv = getImageDescriptorFV(opts, GMM, PCA, descrs)
% -------------------------------------------------------------------------

% Project into PCA space
xy = descrs(opts.SIFTDIM+1:end,:);
descrs=bsxfun(@minus, descrs(1:opts.SIFTDIM,:), PCA.mean);
descrs=PCA.eigvec'*descrs;

descrs = [descrs; xy];

% Extracts FV representation using the GMM
fv  =  vl_fisher(descrs, GMM.mu, GMM.sigma, GMM.we, 'Improved');
end

function X = normFV(X)
% -------------------------------------------------------------------------
X = sign(X).*sqrt(abs(X));
X = bsxfun(@rdivide, X, sqrt(sum(X.*X)));
X(isnan(X)) = 0;
end

function [descrs_normalized,frames_normalized] = normalizeSift(opts,descrs,frames)
% -------------------------------------------------------------------------
descrs_normalized = descrs;

xy = descrs_normalized(opts.SIFTDIM+1:end,:);
descrs_normalized = descrs_normalized(1:opts.SIFTDIM,:);

% Remove empty ones
idx = find(sum(descrs_normalized)==0);
descrs_normalized(:,idx)=[];
if nargin < 3
    frames_normalized = [];
else
    frames_normalized = frames;
    frames_normalized(:,idx) = [];
end

% Square root:
descrs_normalized = sqrt(descrs_normalized);

% 1/4 norm
X = sum(descrs_normalized.*descrs_normalized).^-0.25;
descrs_normalized = bsxfun(@times, descrs_normalized,X);

xy(:,idx) = [];
descrs_normalized = [descrs_normalized; xy];

descrs_normalized(isnan(descrs_normalized))=0;
end

function im = adjustImage(im)
imOrig = im;
im = im2bw(im);
[h,w] = size(im);
x = find(im==0);
w1 = ceil(min(x)/h);
w2 = floor(max(x)/h);
h1 = min(mod(x,h))+1;
h2 = max(mod(x,h))-1;
im = imOrig(h1:h2,w1:w2);
end
