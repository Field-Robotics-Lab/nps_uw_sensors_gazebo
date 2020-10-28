clear;clc;

nBeams = 17;

opts = delimitedTextImportOptions("NumVariables", 26);
opts.DataLines = [4, Inf];opts.Delimiter = ",";
opts.ExtraColumnsRule = "ignore";opts.EmptyLineRule = "read";
Data = readtable("/tmp/SonarRawData_000001.csv", opts);
clear opts

figure;
plotSkips = 1;
nPlots = length(1:plotSkips:nBeams);
distance = (1:length(Data{:,1}))/1500;
for i=1:nPlots
    for j=1:length(Data{:,1})
        temp(j) = str2num(cell2mat(Data{j,i}));
    end
    temp2 = ifft(temp);
    
    subplot(1,nPlots,i);
    semilogx(abs(temp2),distance);
end