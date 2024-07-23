%%
clear all ;close all;clc
%% load the data
% ID = 567;
% subj = sprintf('%d_train_Data.mat', ID);
% load(subj);
%% Trial rejection based on the mean energy
% Define the folder containing the train data files
folder_path = "D:\Master's Project\preprocessing\Train_Data\dep";

% List all the files in the folder
file_list = dir(fullfile(folder_path, '*.mat'));
% desired channels
desired_channels = 1:57;
desired_channels = [desired_channels,59, 60,61];
for s = 1:length(file_list)
    % Load the feature file
    file_name = fullfile(folder_path, file_list(s).name);
    disp(file_name)
    subj = load(file_name);
    ID = subj.Train_Data.ID;
    
    valid_trials = subj.Train_Data.Total.valid_trials;
    Energy_all = zeros(length(valid_trials), length(desired_channels));
    for i = 1:length(valid_trials)
        for j = 1:length(desired_channels)
            chan = desired_channels(j);
            % start: the start of the trial
            start_trial = subj.Train_Data.Total.events(3, valid_trials(i));
            % stop:  the end of trial
            stop_trial = subj.Train_Data.Total.events(4, valid_trials(i));
            segment = subj.Train_Data.Total.data(chan, start_trial:stop_trial);
            segment = double(segment);
            energy = sum(segment.^2)/length(segment);
            Energy_all(i, j) = energy;
        end
    end
    trial_mean_energy = mean(Energy_all,2);
    sorted_energy = sort(trial_mean_energy);
    % figure;
    % subplot(2,1,1); plot(trial_mean_energy); xlabel('Trial'); title('Trials mean energy');
    % subplot(2,1,2); plot(sorted_energy); title('Sotred trials mean energy');
    % sgtitle('567, depressed');
    %

    low_tail = ceil(0.05*length(sorted_energy));
    up_tail = floor(0.95*length(sorted_energy));
    truncated_energy = sorted_energy(low_tail:up_tail);
    %
    med_energy = median(truncated_energy);
    std_energy = std(truncated_energy);
    columns_to_remove = [];
    up_band = med_energy + 2*std_energy;
    low_band = med_energy - 2*std_energy;
    for k = 1:size(trial_mean_energy,1)
        if (trial_mean_energy(k)> up_band)||(trial_mean_energy(k) < low_band)
            columns_to_remove = [columns_to_remove,k];
        end 
    end
    %
    % upper_band = up_band * ones(1,length(valid_trials));
    % lower_band = low_band * ones(1,length(valid_trials));
    % figure
    % plot(trial_mean_energy)
    % xlabel("Trial");
    % ylabel('Trial mean energy');
    % title('567, depressed')
    % hold on
    % plot(upper_band)
    % plot(lower_band)
    %
    % remove bad trials from valid trials
    valid_trials(columns_to_remove) = [];
    % structing
    subj.Train_Data.Total.valid_trials = valid_trials;
    clear Energy_all energy trial_mean_energy up_band low_band std_energy med_energy stop_trial start_trial...
        columns_to_remove chan i j low_tail up_tail truncated_energy sorted_energy
    % save the trial rejection train_Data
    filename = sprintf('%d_TR_Train_Data.mat', ID);

    % Save the struct to a file
    save(filename, 'subj');
    clear Train_Data
end