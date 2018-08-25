function [cutoff,fixstart,fixend,starttime,endtime,fixdur,xmedian,ymedian,flankdataloss,fracinterped] = getFixations(finalweights,timestamp,xpos,ypos,missing,cutoffstd,maxMergeDist,maxMergeTime,minFixDur)
% determine fixations based on finalweights from 2-means clustering

% Roy Hessels - 2014

%% input:

% finalweights              = 2-means clustering weighting
% timestamp                 = timestamp from ET (should be in ms!)
% xpos, ypos                = horizontal & vertical coordinates from ET
% missing                   = boolean vector indicating which samples are
%                             missing (originally, before interpolation!)
% cutoffstd                 = number of std above mean clustering-weight to
%                               use as fixation cutoff
% maxMergeDist              = maximum Euclidean distance in pixels between fixations for merging
% maxMergeTime              = maximum time in ms between fixations for merging

%% output:

% cutoff                    = cutoff used for fixation detection
% fixstart                  = vector with fixation start indices
% fixend                    = vector with fixation end indices
% starttime                 = vector with fixation start times
% endtime                   = vector with fixation end times
% fixdur                    = vector with fixation durations
% xmedian                   = vector with fixation median horizontal
%                               position (one value for each fixation in
%                               trial)
% ymedian                   = vector with fixation median vertical
%                               position (one value for each fixation in
%                               trial)
% flankdataloss             = boolean with 1 for when fixation is flanked
%                               by data loss, 0 if not flanked by data loss
% fracinterped              = fraction of data loss/interpolated data
%                               during fixation

%% first determine cutoff for finalweights
cutoff = nanmean(finalweights) + cutoffstd*nanstd(finalweights);

% get boolean of fixations
fixbool = finalweights < cutoff;

% get indices of where fixations start and end
[fixstart,fixend] = bool2bounds(fixbool);

% get start time, end time, and fix duration
starttime   = timestamp(fixstart);
endtime     = timestamp(fixend);

%% loop over all fixation candidates in trial, see if should be merged
for p=length(starttime):-1:2
    % get median coordinates of fixation
    xmedThis = median(xpos(fixstart(p):fixend(p)));
    ymedThis = median(ypos(fixstart(p):fixend(p)));
    xmedPrev = median(xpos(fixstart(p-1):fixend(p-1)));
    ymedPrev = median(ypos(fixstart(p-1):fixend(p-1)));
    % check if fixations close enough in time and space and thus qualify
    % for merging
    % The interval between the two fixations is calculated correctly (see
    % notes about fixation duration below), i checked this carefully. (Both
    % start and end of the interval are shifted by one sample in time, but
    % assuming practicalyl constant sample interval, thats not an issue.)
    if starttime(p)-endtime(p-1)<maxMergeTime && hypot(xmedThis-xmedPrev,ymedThis-ymedPrev) < maxMergeDist
        % merge
        fixend(p-1) = fixend(p);
        endtime(p-1)= endtime(p);
        % delete merged fixation
        fixstart(p) = [];
        fixend(p)   = [];
        starttime(p)= [];
        endtime(p)  = [];
    end
end

%% beginning and end of fixation must be real data, not interpolated.
% If interpolated, those bit(s) at the edge(s) is excluded from the
% fixation first throw out fixations that are all missing/interpolated
for p=length(starttime):-1:1
    if all(missing(fixstart(p):fixend(p)))
        fixstart(p) = [];
        fixend(p)   = [];
        starttime(p)= [];
        endtime(p)  = [];
    end
end
% then check edges and shrink if needed
for p=1:length(starttime)
    if missing(fixstart(p))
        fixstart(p) = fixstart(p)+find(~missing(fixstart(p):   fixend(p)  ),1)-1;
        starttime(p)= timestamp(fixstart(p));
    end
    if missing(fixend(p))
        fixend(p)   = fixend(p)  -find(~missing(fixend(p)  :-1:fixstart(p)),1)+1;
        endtime(p)  = timestamp(fixend(p));
    end
end

%% caluclate fixation duration
% if you calculate fixation duration by means of time of last sample during
% fixation minus time of first sample during fixation (our fixation markers
% are inclusive), then you always underestimate fixation duration by one
% sample because you're in practice counting to the beginning of the
% sample, not the end of it. To solve this, as end time we need to take the
% timestamp of the sample that is one past the last sample of the fixation.
% so, first get new end time
endtime     = timestamp(min(fixend+1,length(timestamp))); % make sure we don't run off the end of the data
% if last fixation ends at end of data, we need to make up how long that
% sample is and add that to the end time
if ~isempty(fixend) && fixend(end)==length(timestamp)   % first check if there are fixations in the first place, or we'll index into non-existing data
    endtime(end) = endtime(end) + diff(timestamp(end-1:end));
end
% now calculate fixation duration correctly.
fixdur      = endtime-starttime;


%% check if any fixations are too short
qTooShort   = fixdur<minFixDur;
fixstart(qTooShort) = [];
fixend(qTooShort)   = [];
starttime(qTooShort)= [];
endtime(qTooShort)  = [];
fixdur(qTooShort)   = [];

%% process fixations, get other info about them
[   xmedian,...         % vectors for medians
    ymedian,...
    flankdataloss,...   % vectors for whether fixation is flanked by data loss
    fracinterped]      = deal(zeros(size(fixstart)));

for a = 1:length(fixstart)
    idxs = fixstart(a):fixend(a);
    % get data during fixation
    xposf = xpos(idxs);
    yposf = ypos(idxs);
    % for all calculations below we'll only use data that is not
    % interpolated, so only real data
    qMiss = missing(idxs);
    
    % get median coordinates of fixation
    xmedian(a) = median(xposf(~qMiss));
    ymedian(a) = median(yposf(~qMiss));
    
    % determine whether fixation is flanked by period of data loss
    flankdataloss(a) = (fixstart(a)>1 && missing(fixstart(a)-1)) || (fixend(a)<length(xpos) && missing(fixend(a)+1));
    
    % fraction of data loss during fixation that has been (does not count
    % data that is still lost)
    fracinterped(a)  = sum(~isnan(xposf(qMiss)))./(fixend(a)-fixstart(a)+1);
end

return
