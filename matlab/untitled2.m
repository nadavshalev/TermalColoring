
c = normxcorr2(tmag,gmag);
[cy,cx] = size(c);
sz = 50;
c = c(cy/2-sz:cy/2+sz,cx/2-sz:cx/2+sz);
figure, surf(c), shading flat

[max_c, imax] = max(abs(c(:)));
[ypeak, xpeak] = ind2sub(size(c),imax(1));

ymove = sz-ypeak;
xmove = sz-xpeak;

gmag2 = circshift(gmag,ymove);
gmag2 = circshift(gmag2',xmove)';
normGray2 = circshift(normGray,ymove);
normGray2 = circshift(normGray2',xmove)';

figure;imshowpair(tmag.^0.5,gmag2,'falsecolor')
figure;imshowpair(normGray2,normTherm,'falsecolor')