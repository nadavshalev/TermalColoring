close all;clear;clc

path =  '~/Documents/DB/FLIR/';

%%
time = 0;
load('qualityList.mat')
inds = find(qalityList == 3);
for i = inds
    txtnum = sprintf('%05.0f',i);
    imName = ['FLIR_' txtnum '.jpg'];
    thermpath = [path 'Data/FLIR_' txtnum '.tiff'];
    colorpath = [path 'Reg/FLIR_' txtnum '.jpg'];
    thermpath_new = [path 'Calib/TRM/'];
    colorpath_new = [path 'Calib/RGB/'];
    
    copyfile(thermpath,thermpath_new);
    copyfile(colorpath,colorpath_new);
end