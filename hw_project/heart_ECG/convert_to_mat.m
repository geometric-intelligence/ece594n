% MATLAB code to convert all .csv ecg files to .mat (faster loading)

read_folder = './ECGDataDenoised/';
write_folder = './ECGDataDenoisedMat/'; 

all_csv_files = dir(fullfile(read_folder, '*.csv'));
for i = 1:length(all_csv_files)
    file_name_csv = all_csv_files(i).name;
    file_name_mat = strcat(file_name_csv(1:length(file_name_csv)-4), '.mat')
    file_r_path = fullfile(read_folder, file_name_csv)
    file_w_path = fullfile(write_folder, file_name_mat)
    val=csvread(file_r_path);
    save(file_w_path,'val');
    disp(file_name_mat);
    clear val;
    clear file_name_csv;
    clear file_name_mat;
end
