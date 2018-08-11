function [fix,data,par] = I2MCfunc(data,varargin)
% Hessels, R.S., Niehorster, D.C., Kemner, C., & Hooge, I.T.C., (2016).
% Noise-robust fixation detection in eye-movement data - Identification by 
% 2-means clustering (I2MC). Submitted.

%% deal with inputs
% define parser
parser = inputParser;
parser.FunctionName    = mfilename;
parser.KeepUnmatched   = true;
parser.PartialMatching = false;
% required parameters:
parser.addParameter('xres'          , [], @(x) validateattributes(x,{'numeric'},{'scalar'}));
parser.addParameter('yres'          , [], @(x) validateattributes(x,{'numeric'},{'scalar'}));
parser.addParameter('freq'          , [], @(x) validateattributes(x,{'numeric'},{'scalar'}));
parser.addParameter('missingx'      , [], @(x) validateattributes(x,{'numeric'},{'scalar'}));
parser.addParameter('missingy'      , [], @(x) validateattributes(x,{'numeric'},{'scalar'}));
parser.addParameter('scrSz'         , [], @(x) validateattributes(x,{'numeric'},{'numel',2}));
parser.addParameter('disttoscreen'  , [], @(x) validateattributes(x,{'numeric'},{'scalar'}));
% parameters with defaults:
% CUBIC SPLINE INTERPOLATION
% max duration (s) of missing values for interpolation to occur
parser.addParameter('windowtimeInterp'  , 0.1 , @(x) validateattributes(x,{'numeric'},{'scalar'}));
% amount of data (number of samples) at edges needed for interpolation
parser.addParameter('edgeSampInterp'    , 2   , @(x) validateattributes(x,{'numeric'},{'scalar','integer'}));
% (default value set below if needed) maximum displacement during missing for interpolation to be possible
parser.addParameter('maxdisp'           , []  , @(x) validateattributes(x,{'numeric'},{'scalar'}));
% K-MEANS CLUSTERING
% time window (s) over which to calculate 2-means clustering (choose value so that max. 1 saccade can occur)
parser.addParameter('windowtime'        , 0.2 , @(x) validateattributes(x,{'numeric'},{'scalar'}));
% time window shift (s) for each iteration. Use zero for sample by sample processing
parser.addParameter('steptime'          , 0.02, @(x) validateattributes(x,{'numeric'},{'scalar'}));
% downsample levels (can be empty)
parser.addParameter('downsamples'       , [2 5 10], @(x) validateattributes(x,{'numeric'},{'integer'}));
% order of cheby1 Chebyshev downsampling filter, default is normally ok, as
% long as there are 25 or more samples in the window (you may have less if
% your data is of low sampling rate or your window is small
parser.addParameter('chebyOrder'        , 8   , @(x) validateattributes(x,{'numeric'},{'integer'}));
% maximum number of errors allowed in k-means clustering procedure before proceeding to next file
parser.addParameter('maxerrors'         , 100 , @(x) validateattributes(x,{'numeric'},{'scalar','integer'}));
% FIXATION DETERMINATION
% number of standard deviations above mean k-means weights will be used as fixation cutoff
parser.addParameter('cutoffstd'         , 2   , @(x) validateattributes(x,{'numeric'},{'scalar'}));
% maximum Euclidean distance in pixels between fixations for merging
parser.addParameter('maxMergeDist'      , 30  , @(x) validateattributes(x,{'numeric'},{'scalar'}));
% maximum time in ms between fixations for merging
parser.addParameter('maxMergeTime'      , 30  , @(x) validateattributes(x,{'numeric'},{'scalar'}));
% minimum fixation duration (ms) after merging, fixations with shorter duration are removed from output
parser.addParameter('minFixDur'         , 40  , @(x) validateattributes(x,{'numeric'},{'scalar'}));

% get inputs the user specified and throw them in the parser
if isstruct(varargin{1})
    % convert to key-value pairs
    assert(isscalar(varargin),'only one input for options is expected if options are given as a struct')
    varargin = [reshape([fieldnames(varargin{1}) struct2cell(varargin{1})].',1,[]) varargin(2:end)];
end
parse(parser,varargin{:});
par = parser.Results;

% deal nicely with unmatched
unmatched = fieldnames(parser.Unmatched);
if ~isempty(unmatched)
    msg = sprintf('Some parameters were unrecognized:\n');
    for q=1:length(unmatched)
        msg = [msg sprintf('  %s: %s\n',unmatched{q},Var2Str(parser.Unmatched.(unmatched{q})))];
    end
    msg = [msg sprintf('\nValid recognizable parameters are:\n  ')];
    msg = [msg strjoin(parser.Parameters,sprintf('\n  '))];
        
    ME = MException(sprintf('%s:InputError',mfilename),msg);
    throwAsCaller(ME);
end

% deal with required options
% if empty, user did not specify these
checkFun = @(opt,str) assert(~isempty(par.(opt)),'I2MCfunc: %s must be specified using the ''%s'' option',str,opt);
checkFun('xres', 'horizontal screen resolution')
checkFun('yres',   'vertical screen resolution')
checkFun('freq', 'tracker sampling rate')
checkFun('missingx', 'value indicating data loss for horizontal position')
checkFun('missingy', 'value indicating data loss for vertical position')
checkFun('scrSz', 'screen size ([x y]) in cm')
checkFun('disttoscreen', 'distance to screen in cm')
% process parameters with defaults
if isempty(par.maxdisp)
    par.maxdisp               = par.xres*0.2*sqrt(2); % maximum displacement during missing for interpolation to be possible
end

% setup visual angle conversion
pixpercm                    = mean([par.xres par.yres]./par.scrSz(:).');
rad2deg                     = @(x) x/pi*180;
degpercm                    = 2*rad2deg(atan(1/(2*par.disttoscreen)));
pixperdeg                   = pixpercm/degpercm;

%% START ALGORITHM

%% PREPARE INPUT DATA
% deal with monocular data, or create average over two eyes
if isfield(data,'left') && ~isfield(data,'right')
    xpos = data.left.X;
    ypos = data.left.Y;
    missing = isnan(data.left.X) | data.left.X==par.missingx | isnan(data.left.Y) | data.left.Y==par.missingy;
    data.left.missing = missing;
    q2Eyes = false;
elseif isfield(data,'right') && ~isfield(data,'left')
    xpos = data.right.X;
    ypos = data.right.Y;
    missing = isnan(data.right.X) | data.right.X==par.missingx | isnan(data.right.Y) | data.right.Y==par.missingy;
    data.right.missing = missing;
    q2Eyes = false;
elseif isfield(data,'average')
    xpos = data.average.X;
    ypos = data.average.Y;
    missing = isnan(data.average.X) | data.average.X==par.missingx | isnan(data.average.Y) | data.average.Y==par.missingy;
    data.average.missing = missing;
    q2Eyes = isfield(data,'right') && isfield(data,'left');
    if q2Eyes
        % we have left and right and average already provided, but we need
        % to get missing in the individual eye signals
        [llmiss, rrmiss] = getMissing(data.left.X,data.right.X,par.missingx,data.left.Y,data.right.Y,par.missingy);
        data.left.missing  = llmiss;
        data.right.missing = rrmiss;
    end
else % we have left and right, average them
    [data.average.X, data.average.Y, missing, llmiss, rrmiss] = averageEyes(data.left.X,data.right.X,par.missingx,data.left.Y,data.right.Y,par.missingy);
    xpos = data.average.X;
    ypos = data.average.Y;
    data.average.missing = missing;
    data.left.missing    = llmiss;
    data.right.missing   = rrmiss;
    q2Eyes = true;
end

%% INTERPOLATION

% get interpolation windows for average and individual eye signals
fprintf('Searching for valid interpolation windows\n');
interpwins   = findInterpWins(xpos, ypos, missing,par.windowtimeInterp,par.edgeSampInterp,par.freq,par.maxdisp);
if q2Eyes
    llinterpwins = findInterpWins(data.left.X  ,data.left.Y  ,llmiss ,par.windowtimeInterp,par.edgeSampInterp,par.freq,par.maxdisp);
    rrinterpwins = findInterpWins(data.right.X ,data.right.Y ,rrmiss ,par.windowtimeInterp,par.edgeSampInterp,par.freq,par.maxdisp);
end

% Use Steffen interpolation and replace values
fprintf('Replace interpolation windows with Steffen interpolation\n');
[xpos,ypos,missingn]= windowedInterpolate(xpos, ypos, missing, interpwins,par.edgeSampInterp);
if q2Eyes
    [llx ,lly ,llmiss]  = windowedInterpolate(data.left.X  ,data.left.Y  ,llmiss ,llinterpwins,par.edgeSampInterp);
    [rrx ,rry ,rrmiss]  = windowedInterpolate(data.right.X ,data.right.Y ,rrmiss ,rrinterpwins,par.edgeSampInterp);
end


if ~q2Eyes
    %% CALCULATE 2-MEANS CLUSTERING FOR SINGLE EYE
    
    % get kmeans-clustering for averaged signal
    fprintf('2-Means clustering started for averaged signal \n');

    [data.finalweights,stopped] = twoClusterWeighting(xpos,ypos,missingn,par.downsamples,par.chebyOrder,par.windowtime,par.steptime,par.freq,par.maxerrors);

    % check whether clustering succeeded
    if stopped
        fprintf('Clustering stopped after exceeding max errors, continuing to next file \n');
        return
    end
    
    %% CALCULATE 2-MEANS CLUSTERING FOR SEPARATE EYES
elseif q2Eyes
    % get kmeans-clustering for left eye signal
    fprintf('2-Means clustering started for left eye signal \n');
    [finalweights_left,stopped] = twoClusterWeighting(llx,lly,llmiss,par.downsamples,par.chebyOrder,par.windowtime,par.steptime,par.freq,par.maxerrors);
    
    % check whether clustering succeeded
    if stopped
        fprintf('Clustering stopped after exceeding max errors, continuing to next file \n');
        return
    end
    
    % get kmeans-clustering for right eye signal
    fprintf('2-Means clustering started for right eye signal \n');
    [finalweights_right,stopped] = twoClusterWeighting(rrx,rry,rrmiss,par.downsamples,par.chebyOrder,par.windowtime,par.steptime,par.freq,par.maxerrors);
    
    % check whether clustering succeeded
    if stopped
        fprintf('Clustering stopped after exceeding max errors, continuing to next file \n');
        return
    end
    
    %% AVERAGE FINALWEIGHTS OVER COMBINED & SEPARATE EYES
    data.finalweights = nanmean([finalweights_left finalweights_right],2);
end

%% DETERMINE FIXATIONS BASED ON FINALWEIGHTS_AVG
fprintf('Determining fixations based on clustering weight mean for averaged signal and separate eyes + 2*std \n')
[fix.cutoff,fix.start,fix.end,fix.startT,fix.endT,fix.dur,fix.xpos,fix.ypos,fix.flankdataloss,fix.fracinterped] = getFixations(data.finalweights,data.time,xpos,ypos,missing,par.cutoffstd,par.maxMergeDist,par.maxMergeTime,par.minFixDur);
[fix.RMSxy,fix.BCEA,fix.fixRangeX,fix.fixRangeY] = getFixStats(xpos,ypos,missing,fix.start,fix.end,pixperdeg);
        
