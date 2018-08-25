function [RMSxy,BCEA,rangeX,rangeY] = getFixStats(xpos,ypos,missing,fstart,fend,pixperdeg)

%% input:

% xpos, ypos                = horizontal & vertical coordinates from ET
% missing                   = boolean vector indicating which samples are
%                             missing (originally, before interpolation!)
% fstart, fend              = fixation start and end indices
% pixperdeg                 = used for transforming pixels on screen to
%                               degrees

%% output:

% RMSxy                     = RMS of fixation (precision)
% BCEA                      = BCEA of fixation (precision)
% fixRangeX                 = max(xpos) - min(xpos) of fixation
% fixRangeY                 = max(ypos) - min(ypos) of fixation

[   RMSxy,...           % vectors for precision measures
    BCEA,...
    rangeX,...
    rangeY]      = deal(zeros(size(fstart)));

for a=1:length(fstart)
    idxs = fstart(a):fend(a);
    % get data during fixation
    xposf = xpos(idxs);
    yposf = ypos(idxs);
    % for all calculations below we'll only use data that is not
    % interpolated, so only real data
    qMiss = missing(idxs);
    
    % calculate RMS
    % since its done with diff, don't just exclude missing and treat
    % resulting as one continuous vector. replace missing with nan first,
    % use left-over values
    xdif = subsasgn(xposf,substruct('()',{qMiss}),nan);
    ydif = subsasgn(yposf,substruct('()',{qMiss}),nan);
    xdif = diff(xdif).^2; xdif(isnan(xdif)) = [];
    ydif = diff(ydif).^2; ydif(isnan(ydif)) = [];
    c    = xdif + ydif; % 2D sample-to-sample displacement value in pixels
    visuelehoeksq = c./pixperdeg^2; % value in degrees visual angle
    RMSxy(a) = sqrt(mean(visuelehoeksq));
    
    % calculate BCEA (Crossland and Rubin 2002 Optometry and Vision Science)
    stdx = std(xposf(~qMiss))/pixperdeg; % value in degrees visual angle
    stdy = std(yposf(~qMiss))/pixperdeg; % value in degrees visual angle
    if length(yposf(~qMiss))<2
        BCEA(a) = nan;
    else
        xx   = corrcoef(xposf(~qMiss),yposf(~qMiss));
        rho  = xx(1,2);
        P    = 0.68; % cumulative probability of area under the multivariate normal
        k    = log(1/(1-P));
        
        BCEA(a) = 2*k*pi*stdx*stdy*sqrt(1-rho.^2);
    end
    
    % calculate max-min of fixation
    if all(qMiss)
        rangeX(a) = nan;
        rangeY(a) = nan;
    else
        rangeX(a) = (max(xposf(~qMiss)) - min(xposf(~qMiss)))/pixperdeg;
        rangeY(a) = (max(yposf(~qMiss)) - min(yposf(~qMiss)))/pixperdeg;
    end
end
