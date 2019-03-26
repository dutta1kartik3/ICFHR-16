function[augFeat] = cnnJaderbergRepwatts(imagesBatch,iniFile, debugVal)

nodeName=char( getHostName( java.net.InetAddress.getLocalHost ) );
disp(['Running on Node:' nodeName]);

addpath(fileparts(mfilename('fullpath')));
ini = IniConfig();
ini.ReadFile(iniFile);
config=readConfigParams(ini);
%add path libraries
addpath([config.libPath 'matconvnet-1.0-beta18/matlab/']);
%addpath([config.libPath 'CVPR2016/matconvnet-master/matlab/']);
addpath([config.libPath 'matconvnet-1.0-beta18/examples/imagenet/']);
vl_setupnn;
%load dataset indices
%load(config.labelDBFile);
net=load(config.preTrainedFile) ; %loading imageNet network
if(isfield(net,'net'))
    net = net.net;
end
%net.layers{16}=[];
%net.layers{19}=[];
%net.layers{21}=[];

%net.layers{18}=[];
%net.layers{21}=[];
%net.layers{23}=[];

%net.layers{end} = struct('type', 'softmax') ;
%net.layers = net.layers(~cellfun('isempty',net.layers));

%keyboard
%Uncommment for new models
net=vl_simplenn_tidy(net);
net=cnn_imagenet_deploy(net);
%extract features
cntr = 1;
%% get task ID from 'SGE_TASK_ID' environment variable
%TaskIDStr = getenv('SLURM_ARRAY_TASK_ID');
%StepIDStr = getenv('CUSTOM_ARRAY_STEP_ID');
%fprintf('slurm%s step %s\n',TaskIDStr,StepIDStr);

%keyboard;
%{
if ~isempty(TaskIDStr) && ~isempty(StepIDStr)
    startIdx = (str2double(TaskIDStr)-1) * str2double(StepIDStr) + 1;
    endIdx = startIdx+str2double(StepIDStr)-1;
else
    disp('No task ID specified');
    startIdx = 1;
    endIdx = 1;%size(imagesBatch,2);
end
%}
startIdx = 1;
endIdx = size(imagesBatch,2);
%keyboard

%%COMMENT ME
%list.TSTind = list.TSTind(1:100);
%sutta = [list.TRNind  list.VALind];
%sutta = [list.TRNind];
for i=startIdx:endIdx
    tic
    disp(['WordId:' num2str(i) '/' num2str(endIdx)]);
    wIdx = imagesBatch{i};
    
    try
	%{
        if(config.imgMode==0)
            if(~strcmp(prevImgLoc,list.ALLnames{wIdx}))
                pageImg = imread([config.trainDBPath list.ALLnames{wIdx}]);
                wordImg = pageImg(list.cords(wIdx,3):list.cords(wIdx,4),list.cords(wIdx,1):list.cords(wIdx,2),:);
            else
                wordImg = pageImg(list.cords(wIdx,3):list.cords(wIdx,4),list.cords(wIdx,1):list.cords(wIdx,2),:);
            end
            if(size(wordImg,3)>1)
                wordImg=rgb2gray(wordImg);
            end
            if(~islogical(wordImg) && strcmp(config.feat.featColor,'binary'))
                thres = graythresh(wordImg);
                wordImg = im2bw(wordImg,thres);
            end
            if(strcmp(config.feat.featColor,'gray') && islogical(wordImg))
                error('Img is not gray scale');
            end
        elseif(config.imgMode==1)
	%}
            wordImg = wIdx;
            if(size(wordImg,3)>1)
                wordImg=rgb2gray(wordImg);
            end
            if(~islogical(wordImg) && strcmp(config.feat.featColor,'binary'))
                thres = graythresh(wordImg);
                wordImg = im2bw(wordImg,thres);
            end
            if(strcmp(config.feat.featColor,'gray') && islogical(wordImg))
                error('Img is not gray scale');
            end    
        
        if(isempty(wordImg))
            disp(['Warning!!! : Check Word Cordinates of ' list.uid{wIdx}]);
            continue;
        end
        
        %Trimming on binary image (1-> background, 0-> foreground)
        if(islogical(wordImg))
            [r c] = find(~wordImg);
            wordImg = wordImg(min(r):max(r),min(c):max(c));
        end
        
        %%Normalization
        
        %%Commented ALEXNET
        %Making image to RGB to ALEXNET compatibility
        %if(size(wordImg,3)==1)
        %    wordImg(:,:,2) = wordImg(:,:,1);
        %    wordImg(:,:,3) = wordImg(:,:,1);
        %end
        
        %if(islogical(wordImg))
        %    wordImg = wordImg.*255;
        %end
        
        %Extract features
        %im_ = single(wordImg) ; % note: 255 range
        %im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
        %im_ = im_ - net.normalization.averageImage ;
        
        %%Jaderberg
        wordImg = imresize(wordImg, [48, 128]);
    
        wordImg = single(wordImg);
        s = std(wordImg(:));
        wordImg = wordImg - mean(wordImg(:));
        im_ = wordImg / ((s + 0.0001) / 128.0);

        % run the CNN
        res = vl_simplenn(net, im_) ;
        currFeat = squeeze(res(config.feat.layerId).x);
        if(i==startIdx)
            %Initialization
            %augFeat = zeros(numel(list.TSTind),numel(currFeat));
            augFeat = zeros(endIdx-startIdx+1,numel(currFeat));
            %indMat = zeros(numel(list.TSTind),1);
            indMat = zeros(endIdx-startIdx+1,1);
        end
        
        augFeat(cntr,:) = currFeat(:);
        indMat(cntr) = i;
        cntr = cntr + 1;
        
        disp(['-----------------------']);
        toc
    catch err
        disp(err.message);
        warning(['error: in wordId: ' num2str(i)]);
    end
end
