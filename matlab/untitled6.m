
% I1  = imgradient(normTherm,'prewitt').^0.7;
% I2 = imgradient(normGray2,'prewitt');
I1 = normTherm;
I2 = normGray2;


partSize = [80,60]*2;
partNum = size(I1) ./ partSize;
o1 = ones(1,partNum(1));
o2 = ones(1,partNum(2));
C1 = mat2cell(I1,o1*partSize(1), o2*partSize(2));
C2 = mat2cell(I2,o1*partSize(1), o2*partSize(2));

c = normxcorr2(C1{1,1},C2{1,1});
[cy,cx] = size(c);
sz = [30 20];


ind1 = [];
ind2 = [];
mx = zeros(size(C1));

for i = 1:partNum(1)
    for j = 1:partNum(2)
        c = normxcorr2(C1{i,j},C2{i,j});
        c = c(cy/2-sz(1):cy/2+sz(1),cx/2-sz(2):cx/2+sz(2));
        [max_c, imax] = max(abs(c(:)));
        if max_c > 0.25
            mx(i,j) = max_c;
            [movey movex] = ind2sub(size(c),imax);
            disp([movey movex])
            points = [i*partSize(1)-partSize(1)/2;j*partSize(2)-partSize(2)/2];
            ind1 = [ind1 points];
            ind2 = [ind2 (points - [movey;movex])];
        end
    end
end


% figure, surf(mx), shading flat

% figure;plot(ind1(2,:),ind1(1,:),'.')
% hold on;plot(ind2(2,:),ind2(1,:),'.')

tform = fitgeotrans(ind2',ind1','pwl');
normGray3 = imwarp(normGray,tform);

[gmag3, gdir3] = imgradient(normGray3,'prewitt');

figure;imshowpair(normGray3,normTherm,'falsecolor')

% figure;imshowpair(tmag.^0.7,gmag3,'falsecolor')