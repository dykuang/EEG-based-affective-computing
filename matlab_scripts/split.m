% For spliting and save subject's data from DREAMER
%
%
%
%
clear all;
raw = load('DREAMER.mat');

% write meta info
fileID = fopen('DREAMER_meta.txt','a');
fprintf(fileID,'%6s %6s %6s\n','Subject','Age', 'Gender');

Y = zeros(23,18,3);
for i = 1:23
    fprintf('working with subject %d', i)
    X = zeros(128*60, 18, 14);  % only take the last 1 minutes, (time, session, channels)

    % write meta info
    fprintf(fileID,'%6s %6s %6s\n',[sprintf('%.2d', i), raw.DREAMER.Data{1,i}.Age, raw.DREAMER.Data{1,i}.Gender]);

    for j = 1:18
        baseline = raw.DREAMER.Data{1,i}.EEG.baseline{j,1};
        stimuli = raw.DREAMER.Data{1,i}.EEG.stimuli{j,1};
        
        m = mean(baseline(7808/2-128*5+1:7808/2+128*5,:), 1); % 10 seconds in the middle
        s = std(baseline(7808/2-128*5+1:7808/2+128*5,:), 1);
        %m = mean(baseline(end-128*60+1:end,:), 1); % 10 seconds 
        %s = std(baseline(end-128*60+1:end,:), 1);

        X(:,j,:) = (stimuli(end-128*60+1:end,:) - m)./s;

        % write label info
        Y(i,:,1) = raw.DREAMER.Data{1,i}.ScoreValence;
        Y(i,:,2) = raw.DREAMER.Data{1,i}.ScoreArousal;
        Y(i,:,3) = raw.DREAMER.Data{1,i}.ScoreDominance;

    end
    Xname = sprintf("S%.2d_1min.mat", i);
    save(Xname,"X");
end
fclose(fileID);
save("DREAMER/Labels.mat","Y");

