function data = load_dataset(opts)
disp('* Loading dataset *');

if ~exist(opts.fileData, 'file')
    % Loading dataset images and GT
    if strcmpi(opts.dataset,'IAM')        
        data = load_IAM(opts);
    elseif strcmpi(opts.dataset,'GW')
        data = load_GW(opts);
    elseif strcmpi(opts.dataset,'Bentham')
        data = load_Bentham(opts);
    else
        error('Dataset not supported');
    end
    save(opts.fileData,'data');
else
    load(opts.fileData);
end
    
end
