function [finalweights,stopped] = twoClusterWeighting(xpos,ypos,missing,downsamples,chebyOrder,windowtime,steptime,freq,maxerrors)
% Calculates 2-means cluster weighting for eye-tracking data
%
% Input:
% xpos,ypos                     = horizontal and vertical coordinates from eye-tracker over which to calculate 2-means clustering
% missingn                      = boolean indicating which samples are missing
% downsamples                   = downsample levels used for data (1/no downsampling is always done, don't specify that) 
% chebyOrder                    = order of Chebyshev downsampling filter
% windowtime                    = time window (s) over which to calculate 2-means clustering (choose value so that max. 1 saccade can occur)
% steptime                      = time window (s) in each iteration. Use zero for sample by sample processing
% freq                          = sampling frequency of data
% maxerrors                     = maximum number of errors allowed in k-means clustering procedure before proceeding to next file
%
% Output:
% finalweights                  = vector of 2-means clustering weights (one weight for each sample), the higher, the more likely a saccade happened
% stopped                       = whether maxerrors was reached or not

% Roy Hessels - 2014

% calculate number of samples of the moving window
nrsamples =       round(windowtime/(1/freq));
stepsize  = max(1,round(  steptime/(1/freq)));

% create empty weights vector
totalweights = zeros(length(xpos), 1);
totalweights(missing) = nan;
nrtests      = zeros(length(xpos), 1);

% stopped is always zero, unless maxiterations is exceeded. this
% indicates that file could not be analysed after trying for x iterations
stopped = false;
counterrors = 0;

% check downsample levels
nd = length(downsamples);
assert(~any(mod(freq,downsamples)),'Some of your downsample levels are not divisors of your sampling frequency')

% filter signal. Follow the lead of decimate(), which first runs a
% Chebychev filter as specified below
rip = .05;	% passband ripple in dB
[b,a,idxs] = deal(cell(1,nd));
for p=1:nd
    [b{p},a{p}] = cheby1(chebyOrder, rip, .8/downsamples(p));
    idxs{p}     = fliplr(nrsamples:-downsamples(p):1);
end

% see where are missing in this data, for better running over the data
% below.
[on,off] = bool2bounds(missing);
if ~isempty(on)
    %  merge intervals smaller than nrsamples long
    merge=find(on(2:end)-off(1:end-1)-1<nrsamples);
    for p=fliplr(merge)
        off(p)   = off(p+1);
        off(p+1) = [];
        on (p+1) = [];
    end
    % check if intervals at data start and end are large enough
    if on(1)<nrsamples+1
        % not enough data point before first missing, so exclude them all
        on(1)=1;
    end
    if off(end)>length(xpos)-nrsamples
        % not enough data points after last missing, so exclude them all
        off(end)=length(xpos);
    end
    % start at first non-missing sample if trial starts with missing (or
    % excluded because too short) data
    if on(1)==1
        i=off(1)+1;  % start at first non-missing
    else
        i=1;
    end
else
    i=1;
end
eind = i+nrsamples-1;
while eind<=length(xpos)
    % check if max errors is crossed
    if counterrors > maxerrors
        fprintf('Too many empty clusters encountered, aborting file. \n');
        stopped = true;
        finalweights = NaN;
        return
    end
    
    % select data portion of nrsamples
    idx = i:eind;
    [ll_d,IDL_d] = deal(cell(1,nd+1));
    ll_d{1} = [xpos(idx) ypos(idx)];
    
    % Filter the bit of data we're about to downsample. Then we simply need
    % to select each nth sample where n is the integer factor by which
    % number of samples is reduced. select samples such that they are till
    % end of window
    for p=1:nd
        ll_d{p+1} = filtfilt(b{p},a{p},ll_d{1});
        ll_d{p+1} = ll_d{p+1}(idxs{p},:);
    end
    
    % do 2-means clustering
    try
        IDL_d{1}   = kmeans2(ll_d{1});
        for p=2:nd+1
            IDL_d{p} = kmeans2(ll_d{p});
        end
    catch ER
        if strcmp(ER.identifier,'stats:kmeans:EmptyCluster')
            
            % If an empty cluster error is encountered, try again next
            % iteration. This can occur particularly in long
            % fixations, as the number of clusters there should be 1,
            % but we try to fit 2 to detect a saccade (i.e. 2 fixations)
            
            % visual explanation of empty cluster errors:
            % http://www.ceng.metu.edu.tr/~tcan/ceng465_s1011/Schedule/KMeansEmpty.html
            
            fprintf('Empty cluster error encountered at sample %i. Trying again on next iteration. \n',i);
            counterrors = counterrors + 1;
            continue
        else
            fprintf('Unknown error encountered at sample %i. \n',i);
        end
    end
    
    % detect switches and weight of switch (= 1/number of switches in
    % portion)
    [switches,switchesw] = deal(cell(1,nd+1));
    for p=1:nd+1
        switches{p}   = abs(diff(IDL_d{p}));
        switchesw{p}  = 1/sum(switches{p});
    end
    
    % get nearest samples of switch and add weight
    weighted = [switches{1}*switchesw{1}; 0];
    for p=1:nd
        j = find(switches{p+1})*downsamples(p);
        for o=0:downsamples(p)-1
            weighted(j+o) = weighted(j+o) + switchesw{p+1};
        end
    end
    
    % add to totalweights
    totalweights(idx) = totalweights(idx) + weighted;
    % record how many times each sample was tested
    nrtests(idx) = nrtests(idx) + 1;
    
    
    % update i
    i = i + stepsize;
    eind = eind + stepsize;
    qWhichMiss = (on>=i & on<=eind) | (off>=i & off<=eind);
    if any(qWhichMiss)
        % we have some missing in this window. we don't process windows
        % with missing. Move back if we just skipped some samples, or else
        % skip whole missing and place start of window and first next
        % non-missing.
        if on(qWhichMiss)==eind-stepsize+1
            % continue at first non-missing
            i = off(qWhichMiss)+1;
        else
            % we skipped some points, move window back so that we analyze
            % up to first next missing point
            i = min(on(qWhichMiss))-nrsamples;
        end
        eind = i+nrsamples-1;
    end
    if eind>length(xpos) && eind-stepsize<length(xpos)
        % we just exceeded data bound, but previous eind was before end of
        % data: we have some unprocessed samples. retreat just enough so we
        % process those end samples once
        d       = eind-length(xpos);
        eind    = eind-d;
        i       = i-d;
    end
end

% create final weights
finalweights = totalweights./nrtests;

return
