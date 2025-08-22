function test20240710_skeletonize2
%%% This function will obtain worm's posture from brightfield image series.
%%% Modified from test20221124_straighten_scape7.m
%%%
%%% 2024/7/10 Y.Toyoshima
%%%


%%% parameters
pathImage = 'C:\Users\toyo\Desktop\20240508-185610tdTomato-10mW-1\rec matching10fps_8bit.tif';
radiusVarianceFilter = 2;
radiusDilationFilter = 2;
minSizeObject = 10000;
thrArcLen = 300; % threshold of arc length for truncating centerline 
flagDebug = true;

%%% file path for saving
[strDir,strFilename,~] = fileparts(pathImage);
pathImageDebugBinary    = fullfile(strDir,[strFilename,'_skeletonize_debug_1_binary.tif']);
pathImageDebugObject    = fullfile(strDir,[strFilename,'_skeletonize_debug_2_object.tif']);
pathImageDebugSkeleton  = fullfile(strDir,[strFilename,'_skeletonize_debug_3_skeleton.tif']);
pathImageDebugCurveBF   = fullfile(strDir,[strFilename,'_skeletonize_debug_4_curve_BF.tif']);
pathImageDebugCurveBF2  = fullfile(strDir,[strFilename,'_skeletonize_debug_5_curve_BF_truncated.tif']);
pathSave                = fullfile(strDir,[strFilename,'_skeletonize.mat']);

%%% read image
fprintf('Reading image: %s\n',pathImage);
imBF = tiffreadVolume(pathImage);
numT = size(imBF,3);
% numT = 10; % for test


%%% roughly binarizing
fprintf('Binarizing the image; \n   ')
% seV = strel('disk',radiusVarianceFilter);
seV = strel('square',radiusVarianceFilter*2+1);
% seD = strel('disk',radiusDilationFilter);
seD = strel('square',radiusDilationFilter*2+1);
imVar = uint8(stdfilt(imBF, seV.Neighborhood)); % 1st
imVar = uint8(stdfilt(imVar,seV.Neighborhood)); % 2nd
imVar = imdilate(imVar,seD); % dilation
imBin = imVar >= mean(imVar,'all'); % thresholding by mean
% imBin = imopen(imBin,seD);
% imBin = imerode(imBin,seD);
imBin = imdilate(imBin,seD);
if flagDebug
    imwrite(imBin(:,:,1),pathImageDebugBinary,'Compression','none');
    for t=2:numT
        imwrite(imBin(:,:,t),pathImageDebugBinary,'Compression','none','WriteMode','append');
    end
end

%%% skeletonize
posSkel = cell(numT,1);
numUseCC = zeros(numT,1);
imObj = false(size(imBin));
for t=1:numT
    fprintf('Skeletonize bright field image: frame #%d...\n', t);
    [posSkel{t},numUseCC(t),imObj(:,:,t)] = getSkeleton(imBin(:,:,t),minSizeObject);

    if flagDebug
        imSkel = false(size(imObj,[1,2]));
        imSkel(sub2ind(size(imSkel),posSkel{t}(:,1),posSkel{t}(:,2))) = true;
        if t==1
            imwrite(imSkel,     pathImageDebugSkeleton,'Compression','none');
            imwrite(imObj(:,:,t), pathImageDebugObject,'Compression','none');
        else
            imwrite(imSkel,     pathImageDebugSkeleton,'Compression','none','WriteMode','append');
            imwrite(imObj(:,:,t), pathImageDebugObject,'Compression','none','WriteMode','append');
        end        
    end

end

%%% Tracking and curve fitting
disp('Tracking and curve fitting');
[xNew,yNew] = getCurve(posSkel,imObj);

if flagDebug
    for t=1:numT
        imCurveBf = false(size(imBF,[1,2]));
        imCurveBf(  sub2ind_trim(size(imCurveBf),  xNew{t},yNew{t})) = true;
        if t==1
            imwrite(imCurveBf,  pathImageDebugCurveBF,'Compression','none');
        else
            imwrite(imCurveBf,  pathImageDebugCurveBF,'Compression','none','WriteMode','append');
        end        
    end
end

% save(pathSave);


%%% truncate the curve with thrArcLen
xNew2 = cell(size(xNew));
yNew2 = cell(size(yNew));
for t=1:numT
    xTmp=xNew{t};
    yTmp=yNew{t};
    arclen = cumsum(sqrt(diff(xTmp).^2+diff(yTmp).^2));
    idx=find(arclen<thrArcLen,1,'last');
    xNew2{t}=xNew{t}(1:idx);
    yNew2{t}=yNew{t}(1:idx);

    if flagDebug
        imCurveBf2 = false(size(imBF,[1,2]));
        imCurveBf2(  sub2ind_trim(size(imCurveBf2),  xNew2{t},yNew2{t})) = true;
        if t==1
            imwrite(imCurveBf2,  pathImageDebugCurveBF2,'Compression','none');
        else
            imwrite(imCurveBf2,  pathImageDebugCurveBF2,'Compression','none','WriteMode','append');
        end
    end
end

clear im*;
save(pathSave);

end


%%% getSkeleton function converts a binarized image into its skeleton
function [posSkel,numUseCC,imObj] = getSkeleton(imBin,minSizeObject)

posSkel = [];
imObj = [];

% filling holes on the edges
% imBinPadU = true(size(imBin)+[1,0,0]);
% imBinPadD = true(size(imBin)+[1,0,0]);
% imBinPadL = true(size(imBin)+[0,1,0]);
% imBinPadR = true(size(imBin)+[0,1,0]);
imBinPadU = true(size(imBin)+[1,0]);
imBinPadD = true(size(imBin)+[1,0]);
imBinPadL = true(size(imBin)+[0,1]);
imBinPadR = true(size(imBin)+[0,1]);


imBinPadU(2:end,  :,:) = imBin;
imBinPadD(1:end-1,:,:) = imBin;
imBinPadL(:,2:end,  :) = imBin;
imBinPadR(:,1:end-1,:) = imBin;

imBinPadU = imfill(imBinPadU,8,"holes");
imBinPadD = imfill(imBinPadD,8,"holes");
imBinPadL = imfill(imBinPadL,8,"holes");
imBinPadR = imfill(imBinPadR,8,"holes");

imBin2 = imBinPadU(2:end,:,:) | imBinPadD(1:end-1,:,:) ...
    | imBinPadL(:,2:end,:) | imBinPadR(:,1:end-1,:);


% skeletonization
CC = bwconncomp(imBin2);
numCC = cellfun(@numel,CC.PixelIdxList);
idxUseCC = find(numCC>minSizeObject);
numUseCC = numel(idxUseCC);
if isempty(idxUseCC)
    disp('Error occured during binarization: object not found');
    return;
end
if numel(idxUseCC)>1
    disp('Error occured during binarization: multiple object found; use nearest to the center.');
    mindist = inf(size(idxUseCC));
    for p=1:numel(idxUseCC)
        [tmpx,tmpy] = ind2sub(CC.ImageSize,CC.PixelIdxList{idxUseCC(p)});
        mindist(p) = min(sum(([tmpx,tmpy] - CC.ImageSize/2).^2,2));
    end
    [~,tmpidx] = min(mindist);
    idxUseCC = idxUseCC(tmpidx);
end
imObj = false(CC.ImageSize);
imObj(CC.PixelIdxList{idxUseCC}) = true;
skel = bwmorph(imObj,'thin',inf); % skeletonize by thinning
% skel = bwmorph(imObj,'skel',inf); % skeletonize by skel in bwmorph
% skel = bwskel(imObj); % skeletonize by bwskel

% find edges and connecting order
[xSkel,ySkel] = find(skel);
posSkel = [xSkel,ySkel];
end


function [xNew,yNew] = getCurve(posSkel,imObj)
% track head position of worm and approximating posture by spline curve
% assuming posSkel contains positions in multiple frame

numT = numel(posSkel);

GD = cell(numT,1);
idxEnds = cell(numT,1);
for p=1:numT
    dist = squareform(pdist(posSkel{p}));
    dist(dist>=2)=0;
    GD{p} = graph(dist);
    %     idxEnds{p} = find(degree(GD{p})~=2); % for touching head and tail
    idxEnds{p} = find(degree(GD{p})==1);

end

% pos3 = []; %%% for debug
% for p=1:numel(posSkel)asdf
%     pos3 = cat(1,pos3,cat(2,posSkel{p}(idxEnds{p},:),p*ones(numel(idxEnds{p}),1)));
% end

idxHead = zeros(numT,1);
idxTail = zeros(numT,1);
% pass 1: forward tracking,
for p=1:numT
    %     if (numel(idxEnds{p})==2) % head and tail will be defined easily
    %         [~,idxSort] = sort(sum((posSkel{p}(idxEnds{p},:)-sizIm/2).^2,2)); %for head is centered
    %         idxHead(p) = idxEnds{p}(idxSort(1));
    %         idxTail(p) = idxEnds{p}(idxSort(2));
    %     else % multiple candidates for head and tail
    if p>1 && idxHead(p-1)>0 % head in the previous frame can be used
        tmpPosCurrent = posSkel{p}(idxEnds{p},:);
        [~,minIdxHead] = min(sum((tmpPosCurrent-posSkel{p-1}(idxHead(p-1),:)).^2,2));
        idxHead(p) = idxEnds{p}(minIdxHead);
        %             tmpPosCurrent(minIdxHead,:) = inf; % avoid to use the same point
        %             [~,minIdxTail] = min(sum((tmpPosCurrent-posSkel{p-1}(idxTail(p-1),:)).^2,2)); % use nearest to the previous frame
        [~,minIdxTail] = max(distances(GD{p},idxHead(p),idxEnds{p})); % choose point with longest distance from head
        idxTail(p) = idxEnds{p}(minIdxTail);
    else
        idxHead(p)=idxEnds{p}(1); % assuming the worm direction...
        [~,minIdxTail] = max(distances(GD{p},idxHead(p),idxEnds{p})); % choose point with longest distance from head
        idxTail(p) = idxEnds{p}(minIdxTail);
    end
    %     end
end

% pass 2: reverse tracking
for p=numT:-1:1
    if idxHead(p)==0 && p<numT && idxHead(p+1)>0 % head in the next frame can be used
        tmpPosCurrent = posSkel{p}(idxEnds{p},:);
        [~,minIdxHead] = min(sum((tmpPosCurrent-posSkel{p+1}(idxHead(p+1),:)).^2,2));
        idxHead(p) = idxEnds{p}(minIdxHead);
        tmpPosCurrent(minIdxHead,:) = inf; % avoid to use the same point
        %         [~,minIdxTail] = min(sum((tmpPosCurrent-posSkel{p+1}(idxTail(p+1),:)).^2,2)); % use nearest to the previous frame
        [~,minIdxTail] = max(distances(GD{p},idxHead(p),idxEnds{p})); % choose point with longest distance from head
        idxTail(p) = idxEnds{p}(minIdxTail);
    end
end


xNew = cell(numT,1);
yNew = cell(numT,1);
[dim1,dim2] = size(imObj,[1,2]);
for p=1:numT
    [idxNodes,~,idxEdges] = shortestpath(GD{p},idxHead(p),idxTail(p));
    posCenterLine = posSkel{p}(idxNodes,:);
    arclen = cumsum([0;GD{p}.Edges{idxEdges,'Weight'}]);

    %%% spline approximation
    %%% spline(b,y(:)'/spline(b,eye(length(b)),x(:)'))
    %%% https://jp.mathworks.com/help/curvefit/least-squares-approximation-by-natural-cubic-splines.html
    
    %%% usual version:
    % b = linspace(0,max(arclen),10);
    % c = 0:max(arclen);
    % xNew{p} = spline(b,posCenterLine(:,1)'/spline(b,eye(length(b)),arclen(:)'),c);
    % yNew{p} = spline(b,posCenterLine(:,2)'/spline(b,eye(length(b)),arclen(:)'),c);
     
    %%% extrapolation and trimming
    b = linspace(0,max(arclen),10);
    c = -0.2*max(arclen):max(arclen)*1.2; 
    xTmp = spline(b,posCenterLine(:,1)'/spline(b,eye(length(b)),arclen(:)'),c);
    yTmp = spline(b,posCenterLine(:,2)'/spline(b,eye(length(b)),arclen(:)'),c);
    
    isInside = xTmp>=1 & xTmp<=dim1 & yTmp>=1 & yTmp<=dim2;
    xTmp2 = xTmp(isInside);
    yTmp2 = yTmp(isInside);
    idxTmp2 = sub2ind([dim1,dim2],round(xTmp2),round(yTmp2));
    imObjTmp = imObj(:,:,p);
    isInside2 = imObjTmp(idxTmp2);

    xNew{p} = xTmp2(isInside2);
    yNew{p} = yTmp2(isInside2);

    % plot(xNew,yNew)
end

end


function idx = sub2ind_trim(sizIm,x,y)
pos = [round(x(:)),round(y(:))];
flagValid = all(pos<=sizIm & pos>0,2);
idx = sub2ind(sizIm,pos(flagValid,1),pos(flagValid,2));
end