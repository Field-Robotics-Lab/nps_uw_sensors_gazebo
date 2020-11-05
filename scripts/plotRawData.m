clear;clc; close all;

nBeams = 512;
plotSkips = 1;

Data = csvread("../SonarRawData_000001.csv",3,0);
% Data2 = csvread("../SonarRawData2_000001.csv",3,0);

clearvars Beams dist plotData
iIndex = 0;
[Beams,dist] = ndgrid(1:length(1:plotSkips:nBeams), (100:length(Data(:,1)))/1500);
for i=1:plotSkips:nBeams
    iIndex = iIndex + 1;
    jIndex = 0;
    maxValue = 0;
    for j=100:length(Data(:,1))
        jIndex = jIndex + 1;
        plotData(iIndex,jIndex) = Data(j,i);
    end
end
figure; image(log10(abs(plotData)));colormap(hot);

% view(-15,70);
% xlabel('Frequency [Hz]');ylabel('Velocity [m/s]');zlabel('Twisting angle [rad]');
% zlim([0 0.0002]);xlim([f_sets{1}(150) f_sets{1}(700)]);ylim([5.5 19]);

% figure;
% iPlots = 1:plotSkips:nBeams;
% nPlots = length(1:plotSkips:nBeams);
% distance = (1:length(Data(:,1)))/1500;
% for i=1:nPlots
%     for j=1:length(Data(:,1))
%         temp(j) = Data(j,iPlots(i));
%         temp2(j) = Data2(j,iPlots(i));
%     end
% %     temp = ifft(temp,16384);
%     temp2 = ifft(temp2,16384);
%     subplot(1,nPlots,i);
%     semilogx(abs(temp(1:length(distance))),distance); hold on;
%     semilogx(abs(temp2(1:length(distance))),distance);
% end
% legend('GPU','CPU','Location','South');