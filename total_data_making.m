%%
close all;clear all;clc
%%
eeglab
%% load the data and display some information
clc
ID = 591;
subj = sprintf('%d.set', ID);
EEG = pop_loadset(subj);
disp('Loaded EEG dataset:');
disp(['Filename: ' EEG.filename]);
disp(['Sampling rate: ' num2str(EEG.srate) ' Hz']);
disp(['EEG duration (min): ' num2str(EEG.pnts/(500*60))]);
% condition of subject (from excel file)
clc
load('subjects.mat');

% Find the row corresponding to the ID
row_index = find(Data4Import(:, 1) == ID);

% Extract the next 35 columns of that row
if ~isempty(row_index)
    subject_info = Data4Import(row_index, :);
else
    error('ID not found');
end
Train_Data.info = subject_info;
Train_Data.ID = ID;
clear row_index Data4Import subject_info
% define events' tags
clc
phase_codes = {'245', '244', '243', '242', '241', '240'};
stimulus_codes = {'10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'};
stimulus_labels = {'A/B', 'C/D', 'E/F'};
action_codes = {'keypad1','keypad2'};
feedback_codes = {'94','104'};
% structing the tags
tags.phase_codes = phase_codes;
tags. stimulus_codes = stimulus_codes;
tags. stimulus_labels = stimulus_labels;
tags.action_codes = action_codes;
tags.feedback_codes = feedback_codes;
Train_Data.tags = tags;
clear tags phase_codes stimulus_codes stimulus_labels action_codes feedback_codes
clc
% Finding the start and the end of each phases
phases_ind = [];
for i = 1:size(Train_Data.tags.phase_codes, 2)
    code = Train_Data.tags.phase_codes{i};
    ph_ind = [EEG.event(strcmp({EEG.event.type}, code)).latency];
    if numel(ph_ind) == 2 % Check if ph_ind has exactly 2 values
        phases_ind = [phases_ind; ph_ind];  % Append the indices if events exist for the code
    elseif numel(ph_ind) == 1 
        disp([' Warning: Start or end event do not exist for this phase ' num2str(i)]);
    else
        % If both start and end events don't exist, skip this phase
        disp(['Both start and end events do not exist for phase ' num2str(i)]);
    end
end
% structing
Train_Data.phases_ind = phases_ind;
clear ph_ind code i
% Find the stimuli latency and making event matrix
clc
event_mat = [];
for i = 1:size(Train_Data.tags.stimulus_codes, 2)
    code = Train_Data.tags.stimulus_codes{i};
    % Find events corresponding to the current stimulus code
    event_ind = [EEG.event(strcmp({EEG.event.type}, code)).latency];
    key = str2num(code)*ones(1,length(event_ind)); % key shows the tag of each stimulus
    event_mat = [event_mat, [key;event_ind]];
end

% the result of the above loop is based on the keys, but we want them based...
...on the time of each stimulus , so we sort them based on time 
[~, order] = sort(event_mat(2,:));
event_mat = event_mat(:,order);

for i = 1: size(event_mat,2)
    if event_mat(2,i)> max(phases_ind(:))
        event_mat(:,i) = [];
    end
end
% Add phase values to the event matrix
% Create a row vector with phase values 
phases_values = zeros(1,size(event_mat,2));
        
for i = 1: size(event_mat,2)
    ind = event_mat(2,i);
    if (Train_Data.phases_ind(1,1)<ind)&(ind<Train_Data.phases_ind(1,2))
        phases_values(i) = 1;
    elseif (Train_Data.phases_ind(2,1)<ind)&(ind<Train_Data.phases_ind(2,2))
        phases_values(i) = 2;
    elseif (Train_Data.phases_ind(3,1)<ind)&(ind<Train_Data.phases_ind(3,2))
        phases_values(i) = 3;
    elseif (Train_Data.phases_ind(4,1)<ind)&(ind<Train_Data.phases_ind(4,2))
        phases_values(i) = 4;
    elseif (Train_Data.phases_ind(5,1)<ind)&(ind<Train_Data.phases_ind(5,2))
        phases_values(i) = 5;
    else (Train_Data.phases_ind(6,1)<ind)&(ind<Train_Data.phases_ind(6,2))
        phases_values(i) = 6;
    end
end

% Add this row vector to your existing event_mat
event_mat = [phases_values; event_mat;phases_values ];
clear code key order event_ind k i phases_values ind
clc
% Find the end of each trial and add to event matrix (4th row)
num_trials = size(event_mat,2);
for i= 1:num_trials
    if i<num_trials
        curr_trial = i;
        next_trial = i+1;
        if event_mat(1,curr_trial) == event_mat(1,next_trial) % Both trials are in...
            ...the same phase
            event_mat(4,curr_trial) = event_mat(3,next_trial); % the end of the current trial...
        ... isthe start of the next trial
        else 
            event_mat(4,curr_trial) = phases_ind(event_mat(1,curr_trial),2);
        end
    else 
        event_mat(4,i) = phases_ind(event_mat(1,i),2);
    end
end
% structing the base struct
Train_Data.Total.events = zeros(8,size(event_mat,2));  
Train_Data.Total.events(1:4,:) = event_mat;
clear event_mat curr_trial next_trial
% All train trials , EEG data
Train_Data.Total.data = EEG.data;
% Find actions type and latency and add it to event matrix (row 5&6)
action_matrix = [];
actions_1_ind = [EEG.event(strcmp({EEG.event.type}, Train_Data.tags.action_codes{1})).latency];
actions_2_ind = [EEG.event(strcmp({EEG.event.type}, Train_Data.tags.action_codes{2})).latency];

for i = 1:num_trials 
    
    % Extract action events in each trial length
    actions_1 = actions_1_ind((actions_1_ind < Train_Data.Total.events(4, i))...
        & (actions_1_ind > Train_Data.Total.events(3, i)));
    actions_2 = actions_2_ind((actions_2_ind < Train_Data.Total.events(4, i))...
        & (actions_2_ind > Train_Data.Total.events(3, i))); 
     
    if ~isempty(actions_1) && isempty(actions_2)
        % If there is more than one action event, consider the first action event
        [action_index, ~] = min(actions_1);
        action_code = str2num('1');
        action_info = [action_code;action_index];
        
    elseif isempty(actions_1) && ~isempty(actions_2)
        [action_index, ~] = min(actions_2);
        action_code = str2num('2');
        action_info = [action_code;action_index];
       
    elseif ~isempty(actions_1) && ~isempty(actions_2)
        [action_1_index, ~] = min(actions_1);
        [action_2_index, ~] = min(actions_2);
        if action_1_index < action_2_index
            action_code = str2num('1');
            action_info = [action_code;action_1_index];
        else
            action_code = str2num('2');
            action_info = [action_code;action_2_index];
        end
    else
        action_code = NaN;
        action_info = [action_code;NaN];
        
        
    end
    action_matrix = [action_matrix,action_info];
end

% structing 
Train_Data.Total.events(5:6,:) = action_matrix;
clear action_matrix action_code action_info actions_1_ind actions_2_ind action_index...
    actions_1 actions_2 i
% Find feedback type and latency and add it to event matrix (row 7&8)
feedback_matrix = [];
feedback_1_ind = [EEG.event(strcmp({EEG.event.type}, Train_Data.tags.feedback_codes{1})).latency];
feedback_2_ind = [EEG.event(strcmp({EEG.event.type}, Train_Data.tags.feedback_codes{2})).latency];

for i=1:num_trials
    % Extract feedback events between the current and next stimulus indice
    feedback_1 = feedback_1_ind((feedback_1_ind < Train_Data.Total.events(4, i))...
        & (feedback_1_ind > Train_Data.Total.events(3, i)));
    feedback_2 = feedback_2_ind((feedback_2_ind < Train_Data.Total.events(4, i))...
        & (feedback_2_ind > Train_Data.Total.events(3, i)));
    % If there is more than one feedback event, consider the first feedback event 
    if ~isempty(feedback_1) && isempty(feedback_2)
        [feedback_index, ~] = min(feedback_1);
        feedback_code = str2num('94');
        feedback_info = [feedback_code;feedback_index];
        
    elseif isempty(feedback_1) && ~isempty(feedback_2)
        [feedback_index, ~] = min(feedback_2);
        feedback_code = str2num('104');
        feedback_info = [feedback_code;feedback_index];
        
    elseif ~isempty(feedback_1) && ~isempty(feedback_2)
        [feedback_1_index, ~] = min(feedback_1);
        [feedback_2_index, ~] = min(feedback_2);
        if feedback_1_index < feedback_2_index
            feedback_code = str2num('94');
            feedback_info = [feedback_code;feedback_1_index];
        else
            feedback_code = str2num('104');
            feedback_info = [feedback_code;feedback_2_index];
        end
    else
        feedback_code = NaN;
        feedback_info = [feedback_code;NaN];
        
        
    end
    feedback_matrix = [feedback_matrix,feedback_info];
end
% structing
Train_Data.Total.events(7:8,:) = feedback_matrix;
clear feedback_matrix feedback_code feedback_info feedback_1 feedback_2 i...
    feedback_index feedback_2_ind feedback_1_ind
% Define valid trials based on the validity of their events
valid_trials=[];
for i=1:num_trials
    if ~isnan(Train_Data.Total.events(5, i)) && ~isnan(Train_Data.Total.events(7, i))
        valid_trials = [valid_trials,i];
    end
end
Train_Data.Total.valid_trials = valid_trials;
clear num_trials valid_trials
% save the Train_Data
% Save the Train Data
filename = sprintf('%d_Train_Data.mat', ID);

% Save the struct to a file
save(filename, 'Train_Data');
%
clear Train_Data