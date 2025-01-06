import preprocess_func
import analysis_func_blinking as analysis_func

video = r"C:\Users\hillu\OneDrive\מסמכים\Inbal\analysis\109_110\video.mp4"
audio = r"C:\Users\hillu\OneDrive\מסמכים\Inbal\analysis\109_110\audio.wav"
dir_109 = r"C:\Users\hillu\OneDrive\מסמכים\Inbal\analysis\109_110\109"
dir_110 = r"C:\Users\hillu\OneDrive\מסמכים\Inbal\analysis\109_110\110"
time_stamps_109 = fr"{dir_109}\time_stamps_109.csv"
time_stamps_110 = fr"{dir_110}\time_stamps_110.csv"
EMG_signal_109 = fr"{dir_109}\109 EMG\edf\05062022_1749_109_00_SD.edf"
EMG_signal_110 = fr"{dir_110}\110 EMG\edf\05062022_1749_110_00_SD.edf"

# preprocess_func.align_EMG_signals(EMG_signal_109, EMG_signal_110, time_stamps_109, time_stamps_110)

EMG_blink_trig_109 = fr"{dir_109}\109 EMG\edf\signal_in_blink_trig.csv"
EMG_blink_trig_110 = fr"{dir_110}\110 EMG\edf\signal_in_blink_trig.csv"
EMG_audio1_109 = fr"{dir_109}\109 EMG\edf\signal_in_audio1.csv"
EMG_audio1_110 = fr"{dir_110}\110 EMG\edf\signal_in_audio1.csv"
EMG_between_audio1_blink_109 = fr"{dir_109}\109 EMG\edf\signal_between_audio1 and blink.csv"
EMG_between_audio1_blink_110 = fr"{dir_110}\110 EMG\edf\signal_between_audio1 and blink.csv"
EMG_blink_109 = fr"{dir_109}\109 EMG\edf\signal_in_blink.csv"
EMG_blink_110 = fr"{dir_110}\110 EMG\edf\signal_in_blink.csv"
EMG_between_blink_button_109 = fr"{dir_109}\109 EMG\edf\signal_between_blink and button.csv"
EMG_between_blink_button_110 = fr"{dir_110}\110 EMG\edf\signal_between_blink and button.csv"
EMG_button_109 = fr"{dir_109}\109 EMG\edf\signal_in_button.csv"
EMG_button_110 = fr"{dir_110}\110 EMG\edf\signal_in_button.csv"
EMG_between_button_mirror_109 = fr"{dir_109}\109 EMG\edf\signal_between_button and mirror.csv"
EMG_between_button_mirror_110 = fr"{dir_110}\110 EMG\edf\signal_between_button and mirror.csv"
EMG_mirror_109 = fr"{dir_109}\109 EMG\edf\signal_in_mirror.csv"
EMG_mirror_110 = fr"{dir_110}\110 EMG\edf\signal_in_mirror.csv"
EMG_between_mirror_smile_109 = fr"{dir_109}\109 EMG\edf\signal_between_mirror and smile.csv"
EMG_between_mirror_smile_110 = fr"{dir_110}\110 EMG\edf\signal_between_mirror and smile.csv"
EMG_smile_109 = fr"{dir_109}\109 EMG\edf\signal_in_smile.csv"
EMG_smile_110 = fr"{dir_110}\110 EMG\edf\signal_in_smile.csv"
EMG_between_smile_audio2_109 = fr"{dir_109}\109 EMG\edf\signal_between_smile and audio2.csv"
EMG_between_smile_audio2_110 = fr"{dir_110}\110 EMG\edf\signal_between_smile and audio2.csv"
EMG_audio2_109 = fr"{dir_109}\109 EMG\edf\signal_in_audio2.csv"
EMG_audio2_110 = fr"{dir_110}\110 EMG\edf\signal_in_audio2.csv"
EMG_files_lst_109 = [EMG_blink_trig_109, EMG_audio1_109, EMG_between_audio1_blink_109,
                     EMG_blink_109, EMG_between_blink_button_109, EMG_button_109, EMG_between_button_mirror_109,
                     EMG_mirror_109, EMG_between_mirror_smile_110, EMG_smile_109, EMG_between_smile_audio2_109,EMG_audio2_109]
EMG_files_lst_110 = [EMG_blink_trig_110, EMG_audio1_110, EMG_between_audio1_blink_110,
                     EMG_blink_110, EMG_between_blink_button_110, EMG_button_110, EMG_between_button_mirror_110,
                     EMG_mirror_110, EMG_between_mirror_smile_109, EMG_smile_110, EMG_between_smile_audio2_110, EMG_audio2_110]


# analysis_func.EMG_describe_std_mean_all_tasks_xcorr(EMG_files_lst_109, EMG_files_lst_110)
#
# analysis_func.describe_std_mean_all_tasks_coherence(EMG_files_lst_109, EMG_files_lst_110)

# analysis_func.describe_std_mean_all_tasks_pearson(EMG_files_lst_109, EMG_files_lst_110)
#
# analysis_func.blink_analysis(EMG_files_lst_109, EMG_files_lst_110)


breath_signal_describe_109 = fr"{dir_109}\109 breath\snifferdata_070722_010946.txt"
breath_signal_109 = fr"{dir_109}\109 breath\snifferdata_070722_010946.csv"
breath_signal_describe_110 = fr"{dir_110}\110 breath\snifferdata_070722_122006.txt"
breath_signal_110 = fr"{dir_110}\110 breath\snifferdata_070722_122006.csv"

# preprocess_func.shift_breath(breath_signal_describe_109, breath_signal_109, time_stamps_109)
# preprocess_func.shift_breath(breath_signal_describe_110, breath_signal_110, time_stamps_110)

breath_signal_after_shift_109 = fr"{dir_109}\109 breath\breath_after_shift.csv"
breath_signal_after_shift_110 = fr"{dir_110}\110 breath\breath_after_shift.csv"
breath_time_stamps_109 = fr"{dir_109}\breath time stamps.csv"
breath_time_stamps_110 = fr"{dir_110}\breath time stamps.csv"

# preprocess_func.align_breath(breath_signal_109, breath_signal_110, breath_time_stamps_109, breath_time_stamps_110)

breath_trig_109 = fr"{dir_109}\109 breath\signal_in_blink_trig.csv"
breath_trig_110 = fr"{dir_110}\110 breath\signal_in_blink_trig.csv"
breath_audio1_109 = fr"{dir_109}\109 breath\signal_in_audio1.csv"
breath_audio1_110 = fr"{dir_110}\110 breath\signal_in_audio1.csv"
breath_between_audio1_blink_109 = fr"{dir_109}\109 breath\signal_between_audio1 and blink.csv"
breath_between_audio1_blink_110 = fr"{dir_110}\110 breath\signal_between_audio1 and blink.csv"
breath_blink_109 = fr"{dir_109}\109 breath\signal_in_blink.csv"
breath_blink_110 = fr"{dir_110}\110 breath\signal_in_blink.csv"
breath_between_blink_button_109 = fr"{dir_109}\109 breath\signal_between_blink and button.csv"
breath_between_blink_button_110 = fr"{dir_110}\110 breath\signal_between_blink and button.csv"
breath_button_109 = fr"{dir_109}\109 breath\signal_in_button.csv"
breath_button_110 = fr"{dir_110}\110 breath\signal_in_button.csv"
breath_between_button_mirror_109 = fr"{dir_109}\109 breath\signal_between_button and mirror.csv"
breath_between_button_mirror_110 = fr"{dir_110}\110 breath\signal_between_button and mirror.csv"
breath_mirror_109 = fr"{dir_109}\109 breath\signal_in_mirror.csv"
breath_mirror_110 = fr"{dir_110}\110 breath\signal_in_mirror.csv"
breath_between_mirror_smile_109 = fr"{dir_109}\109 breath\signal_between_mirror and smile.csv"
breath_between_mirror_smile_110 = fr"{dir_110}\110 breath\signal_between_mirror and smile.csv"
breath_smile_109 = fr"{dir_109}\109 breath\signal_in_smile.csv"
breath_smile_110 = fr"{dir_110}\110 breath\signal_in_smile.csv"
breath_between_smile_audio2_109 = fr"{dir_109}\109 breath\signal_between_smile and audio2.csv"
breath_between_smile_audio2_110 = fr"{dir_110}\110 breath\signal_between_smile and audio2.csv"
breath_audio2_109 = fr"{dir_109}\109 breath\signal_in_audio2.csv"
breath_audio2_110 = fr"{dir_110}\110 breath\signal_in_audio2.csv"
breath_files_lst_109 = [breath_trig_109, breath_audio1_109, breath_between_audio1_blink_109,
                     breath_blink_109, breath_between_blink_button_109, breath_button_109, breath_between_button_mirror_109,
                     breath_mirror_109, breath_between_mirror_smile_110, breath_smile_109, breath_between_smile_audio2_109,breath_audio2_109]
breath_files_lst_110 = [breath_trig_110, breath_audio1_110, breath_between_audio1_blink_110,
                     breath_blink_110, breath_between_blink_button_110, breath_button_110, breath_between_button_mirror_110,
                     breath_mirror_110, breath_between_mirror_smile_109, breath_smile_110, breath_between_smile_audio2_110, breath_audio2_110]

# analysis_func.breath_describe_std_mean_all_tasks_xcorr(breath_files_lst_109, breath_files_lst_110)

# analysis_func.breath_describe_std_mean_all_tasks_pearson(breath_files_lst_109, breath_files_lst_110)
#
analysis_func.breath_describe_std_mean_all_tasks_coherence(breath_files_lst_109, breath_files_lst_110)
#
# analysis_func.blink_analysis(EMG_files_lst_109, EMG_files_lst_110)

# fnirs_signal_109 = fr"{dir_109}\109 fNIRS\109_05062022_2031_110.edf"
# fnirs_signal_describe_109 = fr"{dir_109}\109 fNIRS\5-6-2022-19_56_52.oxy_raw"
# fnirs_signal_110 = fr"{dir_110}\110 fNIRS\110_05062022_2027_109.edf"
# fnirs_signal_describe_110 = fr"{dir_110}\110 fNIRS\5-6-2022-19_56_51.oxy_raw"


# HR_signal_109 = fr"{dir_109}\109 heartbeat\2022-06-05 19-29-41.txt"
# HR_signal_110 = fr"{dir_110}\110 heartbeat\2022-06-05 19-29-43.txt"


# preprocess_func.shift_HR(HR_signal_109, time_stamps_109)
# preprocess_func.shift_HR(HR_signal_110, time_stamps_110)
#
#
# preprocess_func.shift_fNIRS(fnirs_signal_describe_109, fnirs_signal_109, time_stamps_109)
# preprocess_func.shift_fNIRS(fnirs_signal_describe_110, fnirs_signal_110, time_stamps_110)
#


# fNIRS_signal_after_shift_109 = fr"{dir_109}\109 fNIRS\fNIRS_after_shift.csv "
# fNIRS_signal_after_shift_110 = fr"{dir_110}\110 fNIRS\fNIRS_after_shift.csv "


# preprocess_func.slice_fNIRS_signal_by_tasks(fNIRS_signal_after_shift_109, time_stamps_109)
# preprocess_func.slice_fNIRS_signal_by_tasks(fNIRS_signal_after_shift_110, time_stamps_110)


# preprocess_func.slice_breath_signal_by_tasks(breath_signal_after_shift_109, time_stamps_109)
# preprocess_func.slice_breath_signal_by_tasks(breath_signal_after_shift_110, time_stamps_110)


# HR_signal_after_shift_109 = fr"{dir_109}\109 heartbeat\HR_after_shift.csv"
# HR_signal_after_shift_110 = fr"{dir_110}\110 heartbeat\HR_after_shift.csv"


# preprocess_func.slice_HR_signal_by_tasks(HR_signal_after_shift_109, time_stamps_109)
# preprocess_func.slice_HR_signal_by_tasks(HR_signal_after_shift_110, time_stamps_110)
