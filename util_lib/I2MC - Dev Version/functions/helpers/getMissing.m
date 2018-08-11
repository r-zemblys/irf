function [qLMiss, qRMiss, qBMiss] = getMissing(lx,rx,missingx,ly,ry,missingy)

% get where the missing is
qLMiss = (lx == missingx | isnan(lx)) & ly == missingy | isnan(ly);
qRMiss = (rx == missingx | isnan(rx)) & ry == missingy | isnan(ry);
qBMiss = qLMiss & qRMiss;
