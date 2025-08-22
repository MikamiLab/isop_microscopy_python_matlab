
function test20240711_posture
% Error frames:
% 293
% 1505
% 1577-1584
% 1624-1641
% 1961-1999

frameError = [293,1505,1577:1584,1624:1641,1961:1999];

load('rec matching10fps_8bit_skeletonize.mat');

numInterp = 100;
xInterp = zeros(numT,numInterp);
yInterp = zeros(numT,numInterp);
for t=1:numT
   xTmp = xNew2{t};
   yTmp = yNew2{t};
   arclen = cumsum(sqrt(diff(xTmp).^2+diff(yTmp).^2));
   vOld = [0,arclen];
   vNew = linspace(0,max(arclen),numInterp);
   xInterp(t,:) = interp1(vOld,xTmp,vNew);
   yInterp(t,:) = interp1(vOld,yTmp,vNew);
end
xDiff = diff(xInterp,1,2);
yDiff = diff(yInterp,1,2);
theta = unwrap(atan2(yDiff,xDiff),[],2);
meanTheta = mean(theta,2);
thetaNormed = bsxfun(@minus,theta,meanTheta);
flagUse = true(1,numT);
flagUse(frameError) = false;
thetaUse = thetaNormed(flagUse,:);
[coeff,score,latent,tsquared,explained,mu] = pca(thetaUse);

figure;
plot(cumsum(explained)); % first 3 components explained >99% variances

dimEigenworm = 3;

eigenworm = coeff(:,1:dimEigenworm); % then the first 3 components are used
amp = score(:,1:dimEigenworm); % amplitudes
ampAll = nan(numT,dimEigenworm);
ampAll(flagUse,:) = amp;

figure;
plot(1:numInterp-1,eigenworm); % show eigenworm by angle

% visualize the shape of each eigenworm
ampTmp = eye(dimEigenworm);
meanThetaTmp = zeros(dimEigenworm,1);
thetaReconstructed = ampTmp*eigenworm' + meanThetaTmp;
tmp0 = zeros(size(thetaReconstructed,1),1);
xPos = [tmp0,cumsum(cos(thetaReconstructed),2)];
yPos = [tmp0,cumsum(sin(thetaReconstructed),2)];
figure; 
clmap = lines(size(xPos,1));
for p=1:size(xPos,1)
    hold on;
    plot(xPos(p,:),yPos(p,:),'-','Color',clmap(p,:));
    plot(xPos(p,1),yPos(p,1),'o','Color',clmap(p,:));
end

% visualize worm's posture dynamics in time series
figure;
plot(ampAll);
legend({'amplitude #1','amplitude #2','amplitude #3'});

% visualize worm's posture dynamics in phase plot
figure; 
plot3(ampAll(:,1),ampAll(:,2),ampAll(:,3));
grid on;
box on;
xlabel('amplitude #1');
ylabel('amplitude #2');
zlabel('amplitude #3');


% smoothing
ampAllFilt = sgolayfilt(ampAll,3,21);

figure;
plot(ampAllFilt);
legend({'ampFilt #1','ampFilt #2','ampFilt #3'});

% visualize worm's posture dynamics in phase plot
figure; 
plot3(ampAllFilt(:,1),ampAllFilt(:,2),ampAllFilt(:,3));
grid on;
box on;
xlabel('ampFilt #1');
ylabel('ampFilt #2');
zlabel('ampFilt #3');

% save(mfilename);




Mdl = rica(thetaUse',dimEigenworm);
icaWormTmp = Mdl.transform(thetaUse');
stdTmp = std(icaWormTmp);
icaWorm = icaWormTmp./stdTmp; % normalized
ampIcaTmp = Mdl.TransformWeights;
ampIca = ampIcaTmp.*stdTmp;

figure; 
imagesc(thetaUse);
crange = clim;
colorbar;
title('thetaUse');

figure;
imagesc(ampIcaTmp*icaWormTmp');
clim(crange);
colorbar;
title('thetaReconstructed');

figure;
imagesc(ampIca*icaWorm');
clim(crange);
colorbar;
title('thetaReconstructed, normalized');

figure;
plot(1:99,icaWorm); % show icaworm by angle

ampTmp = eye(dimEigenworm);
meanThetaTmp = zeros(dimEigenworm,1);
thetaReconstructedIca = ampTmp*icaWorm' + meanThetaTmp;
tmp0 = zeros(size(thetaReconstructedIca,1),1);
xPosIca = [tmp0,cumsum(cos(thetaReconstructedIca),2)];
yPosIca = [tmp0,cumsum(sin(thetaReconstructedIca),2)];
figure; 
clmap = lines(size(xPosIca,1));
for p=1:size(xPosIca,1)
    hold on;
    plot(xPosIca(p,:),yPosIca(p,:),'-','Color',clmap(p,:));
    plot(xPosIca(p,1),yPosIca(p,1),'o','Color',clmap(p,:));
end

% visualize worm's posture dynamics in time series
figure;
plot(ampIca);
legend({'ampIca #1','ampIca #2','ampIca #3'});

% visualize worm's posture dynamics in phase plot
figure; 
plot3(ampIca(:,1),ampIca(:,2),ampIca(:,3));
grid on;
box on;
xlabel('ampIca #1');
ylabel('ampIca #2');
zlabel('ampIca #3');


% smoothing
ampIcaFilt = sgolayfilt(ampIca,3,21);

figure;
plot(ampIcaFilt);
legend({'ampIcaFilt #1','ampIcaFilt #2','ampIcaFilt #3'});

% visualize worm's posture dynamics in phase plot
figure; 
plot3(ampIcaFilt(:,1),ampIcaFilt(:,2),ampIcaFilt(:,3));
grid on;
box on;
xlabel('ampIcaFilt #1');
ylabel('ampIcaFilt #2');
zlabel('ampIcaFilt #3');




%%% tde-rica

% dimIca = 3; 
% dimIca = 5; 
dimIca = 7; % obtained from Ahamed T 2021
dimEmbed = 8; % obtained from Ahamed T 2021; 12 frames / 16 Hz = 0.75
% dimEmbed = 16; 
% dimEmbed = 25; % about a half of head bending cycle
% dimEmbed = 32; 

% for dimIca=[3,5,7]
%     for dimEmbed=[8,16,25,32]


thetaNan = thetaNormed;
thetaNan(~flagUse,:) = nan;
thetaEmbed = delayembed(thetaNan,dimEmbed);
flagUse2 = ~any(isnan(thetaEmbed),[2,3]);
thetaEmbedUse = thetaEmbed(flagUse2,:,:);
thetaEmbedUseFlatten = reshape(thetaEmbedUse,[sum(flagUse2),dimEmbed*(numInterp-1)]);

Mdl2 = rica(thetaEmbedUseFlatten',dimIca);
icaWorm2Tmp = Mdl2.transform(thetaEmbedUseFlatten');
std2Tmp = std(icaWorm2Tmp);
icaWorm2Flatten = icaWorm2Tmp./std2Tmp; % normalized
icaWorm2 = reshape(icaWorm2Flatten,[dimEmbed,numInterp-1,dimIca]);
ampIca2Tmp = Mdl2.TransformWeights;
ampIca2 = ampIca2Tmp.*std2Tmp;
ampIca2Nan = nan(numel(flagUse2),dimIca);
ampIca2Nan(flagUse2,:) = ampIca2;
ampIca2Nan = cat(1,nan(round((dimEmbed-1)/2),dimIca),...
    ampIca2Nan,nan(dimEmbed-1-round((dimEmbed-1)/2),dimIca));

% figure; 
% imagesc(thetaEmbedUseFlatten);
% crange2 = clim;
% colorbar;
% title('thetaEmbedUse');
% 
% figure;
% imagesc(ampIca2Tmp*icaWorm2Tmp');
% clim(crange2);
% colorbar;
% title('thetaEmbedReconstructed');
% 
% figure;
% imagesc(ampIca2*icaWorm2Flatten');
% clim(crange2);
% colorbar;
% title('thetaEmbedReconstructed, normalized');


% figure;
% for p=1:dimIca
%     subplot(dimIca,1,p);
%     imagesc(icaWorm2(:,:,p)); 
%     colorbar;
% end
% 
% figure;
% for p=1:dimIca
%     subplot(dimIca,1,p);
%     plot(ampIca2Nan(:,p)); 
% end
% 
% figure;
% plotmatrix(ampIca2Nan);


%%% behavioral annotation
framesForward = [1:109,133:223,287:1511,1634:1762,1809:1902]; % defined by eye
framesBackward = [110:132,224:286,1512:1554,1763:1808,1902:1945]; % defined by eye
framesCoiled = [1555:1633,1946:1999]; % defined by eye

figure;
for p=1:dimIca
    subplot(dimIca,1,p);
    plot(ampIca2Nan(:,p)); 
    hold on;
    % plot(framesForward, ampIca2Nan(framesForward, p),'b.'); 
    plot(framesBackward,ampIca2Nan(framesBackward,p),'r.'); 
    plot(framesCoiled,  ampIca2Nan(framesCoiled,  p),'g.'); 
    if p==1
        title(sprintf('dimIca=%d, dimEmbed=%d',dimIca,dimEmbed));
    end
end


figure;
[~,AX,BigAx] = plotmatrix(ampIca2Nan);
for p=1:dimIca
    for q=p+1:dimIca
        % subplot(dimIca,dimIca,(p-1)*dimIca+q);
        axes(AX(p,q));
        hold on;
        plot(ampIca2Nan(framesBackward,q),ampIca2Nan(framesBackward,p),'r.')
        plot(ampIca2Nan(framesCoiled,  q),ampIca2Nan(framesCoiled,  p),'g.')
    end
end
axes(BigAx);
title(sprintf('dimIca=%d, dimEmbed=%d',dimIca,dimEmbed));

    % end
% end

save(mfilename);




end




function tcrsEmbed = delayembed(tcrs,dimEmbed)
tcrsEmbed = zeros(size(tcrs,1)-dimEmbed+1,dimEmbed,size(tcrs,2));
for p=1:dimEmbed
    tcrsEmbed(:,p,:) = tcrs(p:end-dimEmbed+p,:);
end
end


function tcrsRaw = delayembed_inv(tcrsEmbed)
[numTEmbed,dimEmbed,numUseCell] = size(tcrsEmbed);
numT = numTEmbed+dimEmbed-1;
tmpTcrs = nan(numT,dimEmbed,numUseCell);
for p=1:dimEmbed
    tmpTcrs(p:end-dimEmbed+p,p,:) = tcrsEmbed(:,p,:);
end
tcrsRaw = permute(mean(tmpTcrs,2,'omitnan'),[1,3,2]);
end

