function [IT,IG, IC] = loadImage(folder)
if nargin < 1
    folder = '../../../data/camTest/';
end

[file,~] = uigetfile([folder '*.png']);
sp = split(file,'_');
last_n = split(sp{4},'.');
fileName = [sp{2} '_' sp{3} '_' last_n{1}];

thermal_name = 'RCKD_';
thermal_ext = '.txt';
color_name = 'img_';
color_ext = '.png';

raw_text_from_file = fileread([folder thermal_name fileName thermal_ext]);

%striping the file from unneeded signs
edited_filetext = strrep(raw_text_from_file,'[','');
edited_filetext = strrep(edited_filetext,']','');
edited_filetext = strrep(edited_filetext,',','');

% convert the string read from the 
[Output_1d_array, ~] = str2num(edited_filetext);

%use function "reshape" to convert a 1d array to a 2d array
thermalIm = transpose(reshape(Output_1d_array, 480, 640));

% figure;
rgbIm = imread([folder color_name fileName color_ext]);

tmin = min(min(thermalIm));
tmax = max(max(thermalIm));
IT = imadjust((thermalIm - tmin) / (tmax-tmin));

normRgb = imresize(rgbIm,1/2.25);
IG = im2double(rgb2gray(normRgb));
IC = im2double(normRgb);
end

