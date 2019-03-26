function[augFeat] = cnnJaderbergAug(imagesBatch,iniFile, debugVal)

nodeName=char( getHostName( java.net.InetAddress.getLocalHost ) );
disp(['Running on Node:' nodeName]);

addpath(fileparts(mfilename('fullpath')));
ini = IniConfig();
ini.ReadFile(iniFile);
config=readConfigParams(ini);
%add path libraries
addpath([config.libPath 'matconvnet-1.0-beta18/matlab/']);
addpath([config.libPath 'matconvnet-1.0-beta18/examples/imagenet/']);
vl_setupnn;

net=load(config.preTrainedFile) ; %loading imageNet network
if(isfield(net,'net'))
    net = net.net;
end

%keyboard
%Uncommment for new models
net=vl_simplenn_tidy(net);
net=cnn_imagenet_deploy(net);

%extract features
cntr = 1;

startIdx = 1;
endIdx = size(imagesBatch,2);
%keyboard

for i=startIdx:endIdx
    tic
    disp(['WordId:' num2str(i) '/' num2str(endIdx)]);
    wIdx = imagesBatch{i};
    
    try
    
        if(size(wIdx,3)>1)
            wIdx=rgb2gray(wIdx);
        end
        if(~islogical(wIdx) && strcmp(config.feat.featColor,'binary'))
            thres = graythresh(wIdx);
            wIdx = im2bw(wIdx,thres);
        end
        if(strcmp(config.feat.featColor,'gray') && islogical(wIdx))
            error('Img is not gray scale');
        end    
    
      if(isempty(wIdx))
            disp(['Warning!!! : Check Word Cordinates of ' list.uid{wIdx}]);
            continue;
      end
    
      %Trimming on binary image (1-> background, 0-> foreground)
      if(islogical(wIdx))
            [r c] = find(~wIdx);
            wIdx = wIdx(min(r):max(r),min(c):max(c));
      end
	
	counter1 = 1;
	counter2 = 1;
	for j=0:36
	    if(j==0)
		wordImg = wIdx;
	    else
        wordImg = wIdx;
		maxVal=max(max(wordImg));
		theta=[-5,-3,-1,1,3,5];
		thres=[-0.5,-0.3,-0.1,0.1,0.3,0.5];
		tform = affine2d([cosd(theta(counter1)) -sind(theta(counter1)) 0; sind(theta(counter1)) cosd(theta(counter1)) 0; 0 0 1]); %rotation
                wordImg = imwarp(wordImg,tform,'FillValues',maxVal);

                tform = affine2d([1 0 0; thres(counter2) 1 0; 0 0 1]);   %horizontal shear
                wordImg = imwarp(wordImg,tform,'FillValues',maxVal);
		

		counter2 = counter2 + 1;
		if(counter2>6)
			counter1 = counter1+1;
			counter2 = 1;
		end
	    end
            
          %%Jaderberg
          wordImg = imresize(wordImg, [48, 128]);    
          wordImg = single(wordImg);
          s = std(wordImg(:));
          wordImg = wordImg - mean(wordImg(:));
          im_ = wordImg / ((s + 0.0001) / 128.0);

          % run the CNN
          res = vl_simplenn(net, im_) ;
	  if(j==0)
          	currFeat = squeeze(res(config.feat.layerId).x);
		
	  else
		currFeat = currFeat + squeeze(res(config.feat.layerId).x);
	  end
	  
	end
        if(i==startIdx)
            %Initialization
            augFeat = zeros(endIdx-startIdx+1,numel(currFeat));
            indMat = zeros(endIdx-startIdx+1,1);
        end
        currFeat = currFeat/37;
	currFeat = currFeat/norm(currFeat);        
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
