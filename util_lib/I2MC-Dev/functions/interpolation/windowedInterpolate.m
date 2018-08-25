function [xpos,ypos,missing] = windowedInterpolate(xpos,ypos,missing,interpwins,edgesamples,qSpline)
% Steffen interpolation over interpolation windows as returned from
% findInterp.m

if nargin<6
    qSpline=false;
end

% interpolate and replace missing values for position
for a=1:size(interpwins,2)
    % make vector of all samples in this window
    outWin      = interpwins(1,a):interpwins(2,a);
    
    % get edge samples: where no missing data was observed
    % also get samples in window where data was observed
    validsamps  = [outWin(1)+(-edgesamples:-1) outWin(~missing(outWin)) outWin(end)+(1:edgesamples)];
    % get valid values: where no missing data was observed
    validx      = xpos(validsamps);
    validy      = ypos(validsamps);
    
    % do Steffen interpolation, update xpos, ypos
    if qSpline
        xpos(outWin)   = interp1(validsamps,validx,outWin,'spline');
        ypos(outWin)   = interp1(validsamps,validy,outWin,'spline');
    else
        xpos(outWin)   = steffenInterp(validsamps,validx,outWin);
        ypos(outWin)   = steffenInterp(validsamps,validy,outWin);
    end
    % update missing: hole is now plugged
    missing(outWin) = false;
end
