
cp = cpselect(normGray2,normTherm);
movingPointsAdjusted = cpcorr(movingPoints,fixedPoints,normGray2,normTherm);
tform = fitgeotrans(movingPointsAdjusted,fixedPoints,'pwl');
normGray3 = imwarp(normGray2,tform);

[gmag3, gdir3] = imgradient(normGray3,'prewitt');

figure;imshowpair(normGray3,normTherm,'falsecolor')

figure;imshowpair(tmag.^0.7,gmag3,'falsecolor')