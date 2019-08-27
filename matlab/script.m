close all;
[IT,IG, IC] = loadImage();
showImage(IT,IG);
%%
[DT, ~] = imgradient(IT,'prewitt');
[DG, ~] = imgradient(IG,'prewitt');
showImage(DT.^0.8,DG.^0.8);
%%
[RIG,tform, value, values, methond] = thRegister(IT,IG, true);
%%
Rfixed = imref2d(size(IT));
Rmoving = imref2d(size(IG));
RIC = imwarp(IC,Rmoving,tform,'OutputView',Rfixed, 'SmoothEdges', true);

showImage(RIC,IT);


%% 
RIG2 = imwarp(IG,tform);
% [RIG3, move] = xcorCalibration(RIG, RIG2(1:640,1:480), true);
showImage(RIG2,RIG);




%%
[optimizer,metric] = imregconfig('Multimodal');
optimizer.MaximumIterations = 300;
% optimizer.InitialRadius = 0.001;
cost = sum(sum(abs(DG.*DT))) / (480*640)

% figure;
% RIG = imregister(IG,IT,'translation',optimizer,metric); 
% [RDG, ~] = imgradient(RIG,'prewitt');
% subplot(1,2,1); imshowpair(RDG,DT.^0.6); title('translation')
% subplot(1,2,2); imshowpair(RIG,IT); 
% costTranslation = sum(sum(abs(RDG.*DT))) / (480*640)
% drawnow;
% 
% figure;
% RIG = imregister(IG,IT,'rigid',optimizer,metric); 
% [RDG, ~] = imgradient(RIG,'prewitt');
% subplot(1,2,1); imshowpair(RDG,DT.^0.6);  title('rigid')
% subplot(1,2,2); imshowpair(RIG,IT); 
% costRigid = sum(sum(abs(RDG.*DT))) / (480*640)
% drawnow;

figure;
[RIG,tform] = imregister2(IG,IT,'similarity',optimizer,metric);
[RDG, ~] = imgradient(RIG,'prewitt');
subplot(1,2,1); imshowpair(RDG,DT.^0.6);  title('similarity')
subplot(1,2,2); imshowpair(RIG,IT); 
costSimilarity = sum(sum(abs(RDG.*DT))) / (480*640)
drawnow;

% figure;
% RIG = imregister(IG,IT,'affine',optimizer,metric); 
% [RDG, ~] = imgradient(RIG,'prewitt');
% subplot(1,2,1); imshowpair(RDG,DT.^0.6);  title('affine')
% subplot(1,2,2); imshowpair(RIG,IT); 
% costAffine = sum(sum(abs(RDG.*DT))) / (480*640)
% drawnow;

%%
[RDGC, move] = xcorCalibration(DT,DG, true);
move
figure; imshowpair(RDGC,DT.^0.6); 
costXcor = sum(sum(abs(RDGC.*DT))) / (480*640)
%%
% detectMSERFeatures
% points1 = detectHarrisFeatures(IT,'MinQuality', 0.01, 'FilterSize', 21);
% points2 = detectHarrisFeatures(IFG,'MinQuality', 0.01, 'FilterSize', 21);
points1 = detectMinEigenFeatures(IT);
points2 = detectMinEigenFeatures(IG);

figure;imshow(IT,[]); hold on;
plot(points1.selectStrongest(347));
figure;imshow(IG,[]); hold on;
plot(points2.selectStrongest(283));

%% 
IG2 = IG;
IFG = medfilt2(IG2, [7 7]);
IFG = 1-IFG;
% IFG = imadjust(IFG,[],[0 0.6],1);
showImage(IT,IFG);
% figure;imhist(IFG)
% figure;imhist(IT)
%%

[features1,valid_points1] = extractFeatures(IT,points1, 'Method', 'AUTO');
[features2,valid_points2] = extractFeatures(IG,points2, 'Method', 'AUTO');

indexPairs = matchFeatures(features1,features2, 'MaxRatio', 0.01);
matchedPoints1 = valid_points1(indexPairs(:,1));
matchedPoints2 = valid_points2(indexPairs(:,2));
figure; showMatchedFeatures(IT,IG,matchedPoints1,matchedPoints2);
legend('matched points 1','matched points 2');
size(indexPairs)

%%
tform = fitgeotrans(valid_points1(indexPairs(:,1)).Location, valid_points1(indexPairs(:,2)).Location,'pwl');
normGray3 = imwarp(IG,tform);
%%

a = zeros(valid_points1.Count, valid_points2.Count);
for i=1:valid_points1.Count
    for j=1:valid_points2.Count
        if dist(valid_points1.Location(i,:),valid_points2.Location(j,:)) < 30
            a(i,j) = sum(features1.Features(i,:) .* features2.Features(j,:))/64/255;
        end
    end
end

[y,x] = ind2sub(size(a),find(a>0.9));
indexPairs = [y,x];
size(indexPairs)
