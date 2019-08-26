
% I1 = normTherm;%(1:100,end-100:end);
% I2 = normGray2;%(1:100,end-100:end);
% I1 = imgradient(I1,'prewitt').^0.5;
% I2 = imgradient(I2,'prewitt');
I2 = I3(1:100,end-100:end);
I1 = normTherm(1:100,end-100:end);
% I1 = tmag;
% I2 = gmag;
% I2 = edge(normGray,'sobel');
% I1 = edge(normTherm,'sobel');
% I1 = normTherm + double(I1)/4;
points1 = detectHarrisFeatures(I1);
points2 = detectHarrisFeatures(I2);
% points1 = detectBRISKFeatures(I1);
% points2 = detectBRISKFeatures(I2);
figure;imshow(I1,[]); hold on;
plot(points1.selectStrongest(283));
figure;imshow(I2,[]); hold on;
plot(points2.selectStrongest(283));

[features1,valid_points1] = extractFeatures(I1,points1);
[features2,valid_points2] = extractFeatures(I2,points2);

indexPairs = matchFeatures(features1,features2);
matchedPoints1 = valid_points1(indexPairs(:,1));
matchedPoints2 = valid_points2(indexPairs(:,2));
figure; showMatchedFeatures(I1,I2,matchedPoints1,matchedPoints2);
legend('matched points 1','matched points 2');
size(indexPairs)