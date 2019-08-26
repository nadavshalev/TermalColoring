function [RIG,tform, values] = thRegister(IT,IG)
[DT, ~] = imgradient(IT,'prewitt');
[DG, ~] = imgradient(IG,'prewitt');
baseVal = norm(DG.*DT);
[optimizer,metric] = imregconfig('Multimodal');
optimizer.MaximumIterations = 300;

[RIG,tform] = imregister2(IG,IT,'translation',optimizer,metric);
[RDG, ~] = imgradient(RIG,'prewitt');
value = norm(RDG.*DT);
values = value;

[tmpIm,tmpForm] = imregister2(IG,IT,'rigid',optimizer,metric);
[tmpDm, ~] = imgradient(tmpIm,'prewitt');
tmpVal = norm(tmpDm.*DT);
values = [values tmpVal];
if tmpVal > value
    value = tmpVal;
    RIG = tmpIm;
    tform = tmpForm;
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

[tmpIm,tmpForm] = imregister2(IG,IT,'rigid',optimizer,metric);
[tmpDm, ~] = imgradient(tmpIm,'prewitt');
tmpVal = norm(tmpDm.*DT);
values = [values tmpVal];
if tmpVal > value
    value = tmpVal;
    RIG = tmpIm;
    tform = tmpForm;
end

if baseVal > value
    value = -1;
end

end

