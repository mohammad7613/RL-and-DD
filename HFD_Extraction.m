clear all
close all
clc
%%
% Define the folder containing the train data files
folder_path = "D:\Master's Project\preprocessing\Train_data_trial_rejection\dep";

% List all the files in the folder
file_list = dir(fullfile(folder_path, '*.mat'));

% Initialize an empty struct to store the features
HFD = struct();
% desired channels
desired_channels = 1:57;
desired_channels = [desired_channels,59, 60,61];
% Loop through each file
for i = 1:length(file_list)
    % Load the feature file
    file_name = fullfile(folder_path, file_list(i).name);
    disp(file_name)
    subject = load(file_name);
    %define some params
    num_trials =size(subject.subj.Train_Data.Total.valid_trials,2);
    num_stages = 10;
    % define desired trials
    desired_trials = zeros(5,num_stages);
    for i=1:num_stages
        mid_trial = ceil(0.5*(ceil(((i-1)*num_trials)/10)+ ceil((i*num_trials)/10)));
        desired_trials(:,i) = subject.subj.Train_Data.Total.valid_trials(mid_trial-2:mid_trial+2);
    end
    
    % Action-locked HFD
    HFD_action = zeros(5,num_stages,length(desired_channels));
    HFD_mat = zeros(5,num_stages); %each column for each stage
    for chan = 1:length(desired_channels)
            for stage = 1:size(desired_trials, 2)
                for trial = 1:size(desired_trials, 1)
                    % stop:  action time
                    stop_trial = subject.subj.Train_Data.Total.events(6, desired_trials(trial,stage));
                    % start:  400 ms before the action
                    start_trial = stop_trial - 200;
                    segment = subject.subj.Train_Data.Total.data(chan, start_trial:stop_trial);
                    segment = double(segment);
                    hfd = higuchi(segment);
                    HFD_mat(trial, stage) = hfd;
                end
            end
            HFD_action(:, :, chan) = HFD_mat;
    end
    HFD.action = HFD_action;
    clear HFD_mat  segment hfd start_trial stop_trial 
    
    % Feedback-locked HFD 
    HFD_feedback = zeros(5,num_stages,length(desired_channels));
    HFD_mat = zeros(5,num_stages); %each column for each stage
    for chan = 1:length(desired_channels)
            for stage = 1:size(desired_trials, 2)
                for trial = 1:size(desired_trials, 1)
                    % start:  the feedback action
                    start_trial = subject.subj.Train_Data.Total.events(8, desired_trials(trial,stage));
                    % stop:  400 ms after feedback time
                    stop_trial = start_trial + 200;
                    segment = subject.subj.Train_Data.Total.data(chan, start_trial:stop_trial);
                    segment = double(segment);
                    hfd = higuchi(segment);
                    HFD_mat(trial, stage) = hfd;
                end
            end
            HFD_feedback(:, :, chan) = HFD_mat;
    end
    HFD.feedback = HFD_feedback;
    clear HFD_mat segment  hfd start_trial stop_trial
    
        % The whole trial HFD 
    HFD_total = zeros(5,num_stages,length(desired_channels));
    HFD_mat = zeros(5,num_stages); %each column for each stage
    for chan = 1:length(desired_channels)
            for stage = 1:size(desired_trials, 2)
                for trial = 1:size(desired_trials, 1)
                    % start:  the start of trial
                    start_trial = subject.subj.Train_Data.Total.events(3, desired_trials(trial,stage));
                    % stop:  the end of trial
                    stop_trial = subject.subj.Train_Data.Total.events(4, desired_trials(trial,stage));
                    segment = subject.subj.Train_Data.Total.data(chan, start_trial:stop_trial);
                    segment = double(segment);
                    hfd = higuchi(segment);
                    HFD_mat(trial, stage) = hfd;
                end
            end
            HFD_total(:, :, chan) = HFD_mat;
    end
    HFD.total = HFD_total;
        % Keep the subject info
    HFD.info = subject.subj.Train_Data.info;
    HFD.ID = subject.subj.Train_Data.ID;
    clear HFD_mat segment  hfd start_trial stop_trial
    % Save the Train Data
    filename = sprintf('%d_HFD.mat', subject.subj.Train_Data.ID);

    % Save the struct to a file
    save(filename, 'HFD');
end
%%
figure;
for chan = 1:60
HFD_chan = HFD_total(:,:,chan);
mean_chan = mean(HFD_chan);
subplot(10,6,chan);
plot(mean_chan)
end
%%
mean_HFD_total = mean(HFD_action,3);
mean_HFD_total = mean(mean_HFD_total);
plot(mean_HFD_total)
title('Mean HFD for all channels, 567,dep,new method');
%%
function fd = higuchi(x)
% Input:    x       data (1D array)
% Output:   fd      fractal dimension estimate (using Higuchi method)

% Check if input is valid
assert(isvector(x), 'Input must be a 1D array');

% Set-up
N = length(x);          % Get length of signal
kmax = 5;               % Maximal degree of reduction/degree of time stretch (increasing this will increase FD estimates)

% Initialize matrices
Lmk = zeros(kmax, kmax);

% Compute the mean length of curve
for k = 1:kmax
    for m = 1:k
        Lmki = 0;
        % Compute Lmki
        idx = m + (0:fix((N - m) / k)) * k;  % Indexing for faster computation
        Lmki = sum(abs(diff(x(idx))));
        % Compute Ng and update Lmk
        Ng = (N - 1) / (fix((N - m) / k) * k + eps);  % Handle division by zero
        Lmk(m, k) = (Lmki * Ng) / k;
    end
end

% Compute Lk
Lk = sum(Lmk, 1) ./ (1:kmax);

% Calculate the logarithmic values for slope calculation (which is the FD estimate)
lnLk = log(Lk);
lnk = log(1 ./ (1:kmax));

% Calculate the slope and assign it to output
b = polyfit(lnk, lnLk, 1);
fd = b(1);
end  % End of function
