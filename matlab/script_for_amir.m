
fileName = '20082019_150305';
folder = '../../../data/FLIR/training/RGB/';
thermal_name = 'RCKD_Cats_';
thermal_ext = '.txt';
color_name = 'img_Cats_';
color_ext = '.png';

raw_text_from_file = fileread('../../../data/FLIR/training/RGB/FLIR_00011.jpg');

%striping the file from unneeded signs
edited_filetext = strrep(raw_text_from_file,'[','');
edited_filetext = strrep(edited_filetext,']','');
edited_filetext = strrep(edited_filetext,',','');

% convert the string read from the 
[Output_1d_array, ~] = str2num(edited_filetext);

%use function "reshape" to convert a 1d array to a 2d array
thermalIm = transpose(reshape(Output_1d_array, 480, 640));

%use function "imshow" to show the matrix
figure;imshow(thermalIm,[]); impixelinfo;

% figure;
rgbIm = imread('../../../data/FLIR/training/RGB/FLIR_00011.jpg');

tmin = min(min(thermalIm));
tmax = max(max(thermalIm));
normTherm = (thermalIm - tmin) / (tmax-tmin);
MedianFiltered = medfilt2(normTherm,[10 10]);
figure;imshow(MedianFiltered,[]); impixelinfo;

normRgb = imresize(rgbIm,1/2.25);
normGray = rgb2gray(normRgb);

figure;imshow(normGray,[]); 

figure;imshowpair(normRgb,normTherm,'falsecolor')

[tmag, tdir] = imgradient(normTherm,'prewitt');
% figure;imshow(tmag,[]);
[gmag, gdir] = imgradient(normGray,'prewitt');
% figure;imshow(gmag,[]);
figure;imshowpair(tmag,gmag,'falsecolor')

