clear variables; clear mex; close all; fclose('all'); clc;
%dbstop if error;
commandwindow;

opt.xres                        = 1920; % maximum value of horizontal resolution in pixels
opt.yres                        = 1080; % maximum value of vertical resolution in pixels
opt.missingx                    = -opt.xres; % missing value for horizontal position in eye-tracking data (example data uses -xres). used throughout functions as signal for data loss
opt.missingy                    = -opt.yres; % missing value for vertical position in eye-tracking data (example data uses -yres). used throughout functions as signal for data loss


opt.scrSz                       = [53.3 30.1]; % screen size in cm
opt.disttoscreen                = 56.5; % distance to screen in cm.
opt.steptime                    = 0;

folders.data                    = '..\..\etdata\lookAtPoint_EL_irf'; % folder in which data is stored (each folder in folders.data is considered 1 subject)
folders.output                  = 'output'; % folder for output (will use structure in folders.data for saving output)

folders.func                = 'functions'; % folder for functions, will add to matlab path
addpath(genpath(folders.func));

%[file,nfile] = FileFromFolder(fullfile(folders.data),'silent','mat');
file = rdir(fullfile(folders.data, '\**\*i2mc_raw.mat'));
nfile = size(file)

for f = 1:nfile
    

    %% IMPORT DATA
    fprintf('Importing and processing %s; %d out of %d \n',file(f).name, f, nfile)
    tic()
    fname = file(f).name;
    sname = strrep(fname, 'i2mc_raw', 'i2mc');
    data=load(fname);

    opt.freq = double(data.fs);
    if opt.freq == 120
        opt.downsamples=[2 3 5];
        opt.chebyOrder = 7;
    elseif opt.freq == 60
        opt.downsamples=[2 3 4];
        opt.chebyOrder = 3;
    elseif opt.freq == 30
        opt.downsamples=[2 3];
        opt.chebyOrder = 1;
    else
        opt.downsamples=[2 5 10];
        opt.chebyOrder = 8;
    end
    [fix, finalweights]         = I2MCfunc(data,opt);
    toc()
    finalweights=finalweights.finalweights;
    save(sname, 'finalweights')
    delete(fname)
    
    %pname =sprintf('%s/%s_i2mc.png', folders.output, fname(1:end-4));
    %plotResults(data,fix,pname,[opt.xres opt.yres]);
    %return
    close all
end
%stop


