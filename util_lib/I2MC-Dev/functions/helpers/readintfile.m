function [list] = readintfile(name, nskip, cols)

% This function reads tab-delimited textfiles. Numbers are converted to
% float. It will not work if anything else than numbers, NaN or Inf are
% included in the file.

fid         = fopen(name,'rt');
for p=1:nskip
    fgetl(fid);
end
str         = fread(fid,inf,'*char');
st          = fclose(fid);
list        = sscanf(str','%f');

if nargin>=3
    assert(mod(length(list),cols)==0,'Number of columns in file not as expected or file not complete. Got %d elements from file',length(list))
    list = reshape(list,cols,length(list)/cols).';
end