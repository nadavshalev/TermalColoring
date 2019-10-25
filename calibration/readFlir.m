function [IT, IC, IG, IRC, thermpath, colorpath] = readFlir(imnum, path)

txtnum = sprintf('%05.0f',imnum);

if nargin < 2
    path =  '~/Documents/DB/FLIR/';
end
thermpath = [path 'Data/FLIR_' txtnum '.tiff'];
colorpath = [path 'RGB/FLIR_' txtnum '.jpg'];
regpath = [path 'Reg/FLIR_' txtnum '.jpg'];

IRC = [];
if ~exist(thermpath, 'file') || ~exist(colorpath, 'file')
    IT = [];
    IC = [];
    IG = [];
    return;
end

try
    thermalIm = double(imread(thermpath));
    tmin = min(min(thermalIm));
    tmax = max(max(thermalIm));
    IT = (thermalIm - tmin) / (tmax-tmin);
    IC = im2double(imread(colorpath));
    IC = imresize(IC,[512 640]);
    IG = rgb2gray(IC);
    if exist(regpath,'file')
        IRC = imread(regpath);
    end
catch
    IT = [];
    IC = [];
    IG = [];
end


