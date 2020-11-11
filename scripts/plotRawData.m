clear;clc;
% close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nBeams = 512;
FOV = 90;
xPlotRange = 10;
yPlotRange = 5;
filename = "../SonarRawData_000001.csv";
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bw = 29.5e3; % bandwidth
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Data = csvread(filename,4,0); clearvars Beams dist plotData
plotSkips = 1;
iIndex = 0;
[Beams,dist] = ndgrid(1:length(1:plotSkips:nBeams), (100:length(Data(:,1)))/1500);
for i=2:plotSkips:nBeams+1
    iIndex = iIndex + 1;
    jIndex = 0;
    for j=1:length(Data(:,1))
        jIndex = jIndex + 1;
        plotData(iIndex,jIndex) = Data(j,i);
    end
end

delta_t = 1/bw;
vPixelSize = FOV / nBeams;
sonarBeams = (-(FOV/2.0) + ((1:nBeams)-1) * vPixelSize - vPixelSize/2.0);

range_vector = Data(:,1)';
x = range_vector.*cos(sonarBeams'/180*pi);
y = range_vector.*sin(sonarBeams'/180*pi);

figure;
scatterPointSize = 8;
scatter(x(:),y(:),scatterPointSize,20*log10(abs(plotData(:))),'filled')
clims = [-60 0] + 20*log10(max(max(abs(plotData))));
caxis(clims)
colorbar
title(['Bandwidth ' num2str(bw/1000) ' kHz' ])
xlabel('X [m]')
ylabel('Y [m]')
h = colorbar;
ylabel(h,'Echo Level')
axis equal
axis tight
colormap(hot)
set(gca,'Color','k')
xlim(1.02*[0 xPlotRange])
ylim(1.02*[-yPlotRange yPlotRange])

% figure;
% iPlots = 1:30:nBeams;
% nPlots = length(1:30:nBeams);
% for i=2:nPlots-1
%     for j=1:length(Data(:,1))
%         temp(j) = Data(j,iPlots(i));
%     end
%     subplot(1,nPlots-2,i-1);
%     plot(abs(temp(1:length(range_vector))),range_vector);
%     ylim(1.02*[0 xPlotRange])
% end