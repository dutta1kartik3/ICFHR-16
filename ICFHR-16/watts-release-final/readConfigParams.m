function config = readConfigParams(ini)

config = struct;

config.libPath=ini.GetValues('Generic', 'libPath');
config.expId=ini.GetValues('Generic', 'expID');
config.trainDBPath = [ini.GetValues('Generic', 'trainDBPath')];
config.labelDBFile = [ini.GetValues('Generic', 'labelDBFile')];
config.indFolder = [ini.GetValues('Generic', 'expFolder') 'indexing' filesep config.expId filesep];
config.retFolder = [ini.GetValues('Generic', 'expFolder') 'retrieval' filesep config.expId filesep];
config.testCaseFolder = [ini.GetValues('Generic', 'testCaseFolder')];
config.tmpFolder = [ini.GetValues('Generic', 'tmpFolder')];
config.imgMode = [ini.GetValues('Generic', 'imgMode')];
config.preTrainedFile = [ini.GetValues('Generic', 'preTrainedFile')];

config.feat.layerId = ini.GetValues('Features', 'layerId');
config.feat.featColor = ini.GetValues('Features', 'featColor');
config.retrieval.outListSize = ini.GetValues('Retrieval', 'outListSize');
tempString = ini.GetValues('Retrieval', 'distType');
tempVal = textscan(tempString,'%s');
config.retrieval.distType = tempVal{1};
config.retrieval.plotFlag = ini.GetValues('Retrieval', 'plotFlag');
config.retrieval.queryFile = ini.GetValues('Retrieval', 'queryFile');
