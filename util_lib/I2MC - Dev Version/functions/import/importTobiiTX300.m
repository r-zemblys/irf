function [timestamp,lx,ly,rx,ry] = importTobiiTX300(file,nskip,res,missingx,missingy)
% Imports data from Tobii TX300 as return by Tobii SDK (programmed by RH)
% res = [xres yres]

% faster using readintfile (uitgebreid getest, er is geen snellere
% manier)
dat = readintfile(file,nskip,28);

timestamp   = dat(:,28);
lx          = dat(:,8 ) * res(1);
ly          = dat(:,9 ) * res(2);
rx          = dat(:,21) * res(1);
ry          = dat(:,22) * res(2);

% sometimes we have weird peaks where one sample is (very) far outside the
% monitor. Here, count as missing any data that is more than one monitor
% distance outside the monitor.
qMiss = lx<-res(1) | lx>2*res(1) | ly<-res(2) | ly>2*res(2);
lx(qMiss) = missingx;
ly(qMiss) = missingy;
qMiss = rx<-res(1) | rx>2*res(1) | ry<-res(2) | ry>2*res(2);
rx(qMiss) = missingx;
ry(qMiss) = missingy;

return
