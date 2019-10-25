close all;clear;clc
addpath(genpath('./matlab'));
[optimizer,metric] = imregconfig('Multimodal');
optimizer.InitialRadius = 0.0006;
optimizer.Epsilon = 1.5e-6;
optimizer.GrowthFactor = 1.01;
optimizer.MaximumIterations = 3000;

path =  '~/Documents/DB/FLIR/Reg/';

%%
time = 0;
% start = 1;
% endi = 3911;
load('qualityList.mat')
inds = find(qalityList == 2);
for i = inds
    tic
    txtnum = sprintf('%05.0f',i);
    imName = ['FLIR_' txtnum '.jpg'];

    [IT, IC, IG, thermpath, colorpath] = readFlir(i);
    if isempty(IT)
        continue;

    end
    [tmpIm,tmpForm] = imregister2(IG,IT,'affine',optimizer,metric);

    Rfixed = imref2d(size(IT));
    Rmoving = imref2d(size(IG));
    RIC = imwarp(IC,Rmoving,tmpForm,'OutputView',Rfixed, 'SmoothEdges', true);

    imwrite(RIC,[path imName]);
    tmpTime = toc;
    time = time + tmpTime;
%     ava = time/(i-start+1);
%     lft = (endi - i) * ava;
%     disp([num2str(i) '/' num2str(endi) ' - loop: ' num2str(tmpTime) '[sec] avarage: ' num2str(ava) '[sec] left: ' num2str(lft) '[sec]'])
    disp([num2str(i) ': ' num2str(tmpTime)]);
end