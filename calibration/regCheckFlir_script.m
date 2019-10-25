imnum = 3758;

[IT, IC, IG, IRC, thermpath, colorpath] = readFlir(imnum);

ITeq = histeq(IT);
[dT, ~] = imgradient(ITeq,'prewitt');
[dRC, ~] = imgradient(rgb2gray(IRC),'prewitt');
figure; imshowpair(IRC,ITeq); 
figure; imshowpair(dRC,dT);  