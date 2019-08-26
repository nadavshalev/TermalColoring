I1 = normTherm;%(1:100,end-100:end);
I2 = normGray2;%(1:100,end-100:end);


cp = cpselect(I2,I1);
movingPointsAdjusted = cpcorr(movingPoints,fixedPoints,normGray,normTherm);
tform = fitgeotrans(movingPointsAdjusted,fixedPoints,'pwl');
normGray3 = imwarp(normGray,tform);

[gmag3, gdir3] = imgradient(normGray3,'prewitt');

figure;imshowpair(normGray3,normTherm,'falsecolor')

figure;imshowpair(tmag.^0.7,gmag3,'falsecolor')