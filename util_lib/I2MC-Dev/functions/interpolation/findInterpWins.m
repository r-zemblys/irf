function [interpwins] = findInterpWins(xpos,ypos,missing,windowtime,edgesamples,freq,maxdisp)
% find interpolation windows over missingvalue in signal, only if edges of
% window do not exceed a maximum dispersion

% input:
% pos           = position signal (either x or y coordinate)
% windowtime    = time of window to interpolate over in ms
% edgesamples   = number of samples at window edge used for interpolating
%                   in ms
% freq          = frequency of measurement
% missingvalue  = value that corresponds to eye lost in position signal
% maxdisp       = maximum dispersion in position signal (i.e. if signal is
%                   in pixels, provide maxdisp in n pixels)

% output:
% interpwins    = valid indices for interpolation, interpolate using
%                   windowedspline.m

% Roy Hessels - 2014

windowsamples = round(windowtime/(1/freq));


% get indices of where missing intervals start and end
[missStart,missEnd] = bool2bounds(missing);
[dataStart,dataEnd] = bool2bounds(~missing);

% for each candidate, check if have enough valid data at edges to execute
% interpolation. If not, see if merging with adjacent missing is possible
% we don't throw out anything we can't deal with yet, we do that below.
% this is just some preprocessing
k=1;
while k<length(missStart)
    % skip if too long
    if missEnd(k)-missStart(k)+1 > windowsamples
        k = k+1;
        continue;
    end
    % skip if not enough data at left edge
    datk = find(dataEnd==missStart(k)-1,1);
    if dataEnd(datk)-dataStart(datk)+1 < edgesamples
        k = k+1;
        continue;
    end
    % if not enough data at right edge, merge with next. Having not enough
    % on right edge of this one, means not having enough at left edge of
    % next. So both will be excluded always if we don't do anything. So we
    % can just merge without further checks. Its ok if it then grows too
    % long, as we'll just end up excluding that too below, which is what
    % would have happened if we didn't do anything here
    datk = find(dataStart==missEnd(k)+1,1);
    if dataEnd(datk)-dataStart(datk)+1 < edgesamples
        missEnd(k)      = [];
        missStart(k+1)  = [];
        
        % don't advance k so we check this one again and grow it further if
        % needed
        continue;
    end
    
    % nothing left to do, continue to next
    k = k+1;
end

% mark intervals that are too long to be deleted (only delete later so that
% below checks can use all missing on and offsets)
missDur = missEnd-missStart+1;
qRemove = missDur>windowsamples;

% for each candidate, check if have enough valid data at edges to execute
% interpolation and check displacement during missing wasn't too large.
% Mark for later removal as multiple missing close together may otherwise
% be wrongly allowed
for p=1:length(missStart)
    % check enough valid data at edges
    if  missStart(p)<edgesamples+1 ||...                                    % missing too close to beginning of data
        (p>1 && missEnd(p-1) > missStart(p)-edgesamples-1) ||...            % previous missing too close
        missEnd(p)>length(xpos)-edgesamples ||...                           % missing too close to end of data
        (p<length(missStart) && missStart(p+1) < missEnd(p)+edgesamples+1)  % next missing too close
        qRemove(p) = true;
        continue;
    end
    
    % check displacement, per missing interval
    % we want to check per bit of missing, even if multiple bits got merged
    % this as single data points can still anchor where the interpolation
    % goes and we thus need to check distance per bit, not over the whole
    % merged bit
    idx = missStart(p) : missEnd(p);
    [on,off] = bool2bounds(isnan(xpos(idx)));
    for q=1:length(on)
        lesamps = on (q)-[1:edgesamples]+missStart(p)-1;
        resamps = off(q)+[1:edgesamples]+missStart(p)-1;
        if hypot(nanmedian(xpos(resamps))-nanmedian(xpos(lesamps)), nanmedian(ypos(resamps))-nanmedian(ypos(lesamps))) > maxdisp
            qRemove(p) = true;
            break;
        end
    end
    if qRemove(p)
        continue;
    end
end
missStart(qRemove) = [];
missEnd  (qRemove) = [];

% construct windows marking missing data to be interpolated
interpwins = [missStart; missEnd];
