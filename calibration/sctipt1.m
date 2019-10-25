close all;
addpath(genpath('./matlab'));

imnum = 41;

[IT, IC, IG, IRC, thermpath, colorpath] = readFlir(imnum);
if isempty(IT)
    return;
end
ITeq = histeq(IT);
figure; imshowpair(IG,ITeq, 'montage'); 
figure; imshowpair(IG,ITeq); 

%%
[RIG,tform, value, values, baseVal, methond] = thRegister(ITeq, IG, true);

% figure;imshowpair(normG,histeq(IT));

%%
[optimizer,metric] = imregconfig('Multimodal');
optimizer.InitialRadius = 0.001;
optimizer.Epsilon = 1.5e-6;
optimizer.GrowthFactor = 1.01;
optimizer.MaximumIterations = 1000;
[tmpIm,tmpForm] = imregister2(IG,ITeq,'affine',optimizer,metric);

Rfixed = imref2d(size(IT));
Rmoving = imref2d(size(IG));
RIC2 = imwarp(IC,Rmoving,tmpForm,'OutputView',Rfixed, 'SmoothEdges', true);

figure; imshowpair(tmpIm,ITeq); 
figure; imshowpair(RIC2,ITeq); 

[tmpDm, ~] = imgradient(tmpIm,'prewitt');
[tmpDt, ~] = imgradient(ITeq,'prewitt');
figure; imshowpair(tmpDm,tmpDt); 

%%
tmp = circshift(IG, -20);
tmp = circshift(tmp', -30)';
figure; imshowpair(tmp,ITeq, 'montage'); 
figure; imshowpair(tmp,ITeq); 

%%
[DT, ~] = imgradient(IT,'prewitt');
[DG, ~] = imgradient(IG,'prewitt');
% figure; imshowpair(DG,DT);
[RDGC, move] = xcorCalibration(1-ITeq,tmp, true);
figure; imshowpair(RDGC,DT);