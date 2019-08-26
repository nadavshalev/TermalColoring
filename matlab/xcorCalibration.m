function [RIG, move] = xcorCalibration(IT,IG, toPlot)
if nargin == 3
    toPlot = false;
end

c = normxcorr2(IT,IG);
[cy,cx] = size(c);
sz = 50;
c = c(cy/2-sz:cy/2+sz,cx/2-sz:cx/2+sz);

if toPlot
    figure; surf(c); shading flat
end

[~, imax] = max(abs(c(:)));
[ypeak, xpeak] = ind2sub(size(c),imax(1));

ymove = sz-ypeak;
xmove = sz-xpeak;

gmag2 = circshift(IG,ymove);
gmag2 = circshift(gmag2',xmove)';
RIG = circshift(IG,ymove);
RIG = circshift(RIG',xmove)';
move = [ymove xmove]
end

