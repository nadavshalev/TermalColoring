function [RIG,tform, value, values, methond] = thRegister(IT, IG, toPlot)
if nargin < 3
    toPlot = false;
end
methond = ["translation" "rigid" "similarity" "affine"];
[DT, ~] = imgradient(IT,'prewitt');
[DG, ~] = imgradient(IG,'prewitt');
baseVal = norm(DG.*DT);
[optimizer,metric] = imregconfig('Multimodal');
optimizer.MaximumIterations = 300;

[RIG,tform] = imregister2(IG,IT,'translation',optimizer,metric);
[RDG, ~] = imgradient(RIG,'prewitt');
value = norm(RDG.*DT);
values = value;
if toPlot
    figure;
    subplot(1,2,1); imshowpair(RDG,DT.^0.6);  title('translation')
    subplot(1,2,2); imshowpair(RIG,IT); 
    drawnow;
end

[tmpIm,tmpForm] = imregister2(IG,IT,'rigid',optimizer,metric);
[tmpDm, ~] = imgradient(tmpIm,'prewitt');
tmpVal = norm(tmpDm.*DT);
values = [values tmpVal];
if tmpVal > value
    value = tmpVal;
    RIG = tmpIm;
    tform = tmpForm;
end
if toPlot
    figure;
    subplot(1,2,1); imshowpair(tmpDm,DT.^0.6);  title('rigid')
    subplot(1,2,2); imshowpair(tmpIm,IT); 
    drawnow;
end

[tmpIm,tmpForm] = imregister2(IG,IT,'similarity',optimizer,metric);
[tmpDm, ~] = imgradient(tmpIm,'prewitt');
tmpVal = norm(tmpDm.*DT);
values = [values tmpVal];
if tmpVal > value
    value = tmpVal;
    RIG = tmpIm;
    tform = tmpForm;
end
if toPlot
    figure;
    subplot(1,2,1); imshowpair(tmpDm,DT.^0.6);  title('similarity')
    subplot(1,2,2); imshowpair(tmpIm,IT); 
    drawnow;
end


[tmpIm,tmpForm] = imregister2(IG,IT,'affine',optimizer,metric);
[tmpDm, ~] = imgradient(tmpIm,'prewitt');
tmpVal = norm(tmpDm.*DT);
values = [values tmpVal];
if tmpVal > value
    value = tmpVal;
    RIG = tmpIm;
    tform = tmpForm;
end
if toPlot
    figure;
    subplot(1,2,1); imshowpair(tmpDm,DT.^0.6);  title('affine')
    subplot(1,2,2); imshowpair(tmpIm,IT); 
    drawnow;
end

if baseVal > value
    value = -1;
end

end

