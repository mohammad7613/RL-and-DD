# Behavioral Data
This Repository includes the main [EEG Dataset](http://predict.cs.unm.edu/downloads.php)] (d006) related extracted data, more details is provided under each dataset name.

## Subjects_Behavioral_datas

This Dataset is extracted from the EEG recording system tags (attached to each subject EEG .mat file) and the linked measured and extracted score files linked to dataset, consist of BDI score, TAI score, reaction time and accuracy (in average).

"Label" is based on note of the "Data_4_Import.xlsx" file attached to dataset, some more than 13 BDI scored subject had some notes, we categorized them as "MDD" and "Past MDD", Other subjects, are "Healthy" means that their BDI score is less than 7, and "N/A" with BDI more than 13.

"Task" field includes a matrix, describing subject's performance in each stimuli and consequent events, including action and feedback, also I add reaction time which is time difference between onset of stimulation and subject's action of choosing.

Task field has 4 columns, 4 data for each stimuli. Such as,

- Task[0]: stim_ -> Determine type of stimulation, 0 -> AB, 1 -> CD, and 2 -> EF.

- Task[1]: reaction_time_ -> As explained above, the time difference of choosing an action and onset of stimuli.

- Task[2]: action_ -> Determines the action of user, choosing the more valuable Hiragana or the less one, 0 -> The more valuable, 1 -> The less one.

- Task[3]: feedback_ -> Determines whether the subject receive positive feedback (reward, +1) or negative feedback (punishment, -1)
