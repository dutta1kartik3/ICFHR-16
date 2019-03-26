function extract_features(opts)
disp('* Extracting FV features *');
% Extracts the FV representation for every image in the dataset

if  ~exist(opts.fileFeatures,'file') 
    	extract_FV_features_fast(opts);
    
end

end
