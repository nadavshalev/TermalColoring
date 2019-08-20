I2 = normGray;
I1 = normTherm;
I1 = tmag;
I2 = gmag;
points1 = detectHarrisFeatures(tmag);
points2 = detectHarrisFeatures(gmag);
% points1 = detectBRISKFeatures(I1);
% points2 = detectBRISKFeatures(I2);
figure;imshow(normTherm); hold on;
plot(points1.selectStrongest(200));
figure;imshow(normGray); hold on;
plot(points2.selectStrongest(200));

[features1,valid_points1] = extractFeatures(I1,points1);
[features2,valid_points2] = extractFeatures(I2,points2);

indexPairs = matchFeatures(features1,features2);
size(indexPairs)