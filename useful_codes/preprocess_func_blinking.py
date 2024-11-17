import datetime
import os
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import analysis_func_blinking

def get_coherence_interval_blink(blink_window, fs):
    f, t, Sxx = signal.spectrogram(blink_window, fs, return_onesided=False, nperseg=512)
    plt.figure()
    plt.pcolormesh(t, f[np.concatenate((np.arange(11), np.arange(-10, 0)))],
                   Sxx[np.concatenate((np.arange(11), np.arange(-10, 0))), :])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def get_smiles_spectrum(sig1, time_stamps1):
    # a function that was used to calculate the smiles frequencies for coherence
    EMG_fs = 4000
    data1 = mne.io.read_raw_edf(sig1)
    signal1 = (data1.get_data()[5, ])
    start_smile_trig_EMG1, end_smile_trig_EMG1 = get_smile_window_from_EMG(signal1, time_stamps1, EMG_fs)
    smile_window1 = signal1[start_smile_trig_EMG1:end_smile_trig_EMG1] [13000:24500]
    get_coherence_interval_blink(smile_window1, EMG_fs)

def get_breath_spectrum(sig, time_stamps1):
    # a function that was used to calculate the smiles frequencies for coherence
    fs = 20
    data = pd.read_csv(sig)
    signal1 = data.loc[:,'pressure1'].to_numpy()
    start_breath_trig_EMG1, end_breath_trig_EMG1 = get_respiration_window(signal1, time_stamps1, fs)
    breath_window1 = signal1[start_breath_trig_EMG1:end_breath_trig_EMG1]
    f, t, Sxx = signal.spectrogram(breath_window1, fs, return_onesided=False, nperseg=20)
    plt.figure()
    plt.pcolormesh(t, f[np.concatenate((np.arange(5), np.arange(-5, 0)))],
                   Sxx[np.concatenate((np.arange(5), np.arange(-5, 0))), :])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
def align_EMG_signals(sig1, sig2, time_stamps1, time_stamps2):
    """
        align given EMG files and slice and extracts csv files of the signals in each experiment task.
    :param sig1: path to EMG signal of participant1
    :param sig2: path to EMG signal of participant2
    :param time_stamps1: path to csv file with time stamps of them experiment tasks of participant1
    :param time_stamps2: path to csv file with time stamps of them experiment tasks of participant2
    """
    signal1 = read_EEG_signal(sig1)
    signal2 = read_EEG_signal(sig2)
    EMG_fs = 4000
    start_blink_trig_EMG1, end_blink_trig_EMG1 = get_blink_window_from_EMG(signal1, time_stamps1, EMG_fs)
    start_blink_trig_EMG2, end_blink_trig_EMG2 = get_blink_window_from_EMG(signal2, time_stamps2, EMG_fs)
    blink_window1 = signal1[start_blink_trig_EMG1:end_blink_trig_EMG1]
    blink_window2 = signal2[start_blink_trig_EMG2:end_blink_trig_EMG2]
    # get_coherence_interval_blink(blink_window1, EMG_fs)
    blinking_filter1 = create_filter(blink_window1[25000:27000], 4)
    blinking_filter2 = create_filter(blink_window2[24000:27600], 4)
    rms_lst = []
    for i in range(int(EMG_fs/2)):
        blink_window2 = signal2[start_blink_trig_EMG2-i:end_blink_trig_EMG2+i]
        spike_train1 = convert_patten_to_spike_train(blink_window1, blinking_filter1, EMG_fs)
        spike_train2 = convert_patten_to_spike_train(blink_window2, blinking_filter2, EMG_fs)
        rms = calc_RMS(spike_train1, spike_train2, EMG_fs)
        rms_lst.append(rms)
    rms_arr = np.array(rms_lst)
    shift = int(rms_arr.argmin())
    if start_blink_trig_EMG1 > start_blink_trig_EMG2:
        data1 = mne.io.read_raw_edf(sig1)
        start_indx_after_shift = np.abs(start_blink_trig_EMG1-start_blink_trig_EMG2)+shift
        signal1_after_shift = data1.get_data()[0:16,:].T
        slice_EMG_signal_by_tasks(sig2,time_stamps2, EMG_fs)
        slice_EMG_shifted_signal_by_tasks(signal1_after_shift[start_indx_after_shift:,:], time_stamps2, sig1, EMG_fs)
    else:
        data2 = mne.io.read_raw_edf(sig2)
        start_indx_after_shift = np.abs(start_blink_trig_EMG1-start_blink_trig_EMG2)+shift
        signal2_after_shift = data2.get_data()[0:16,:].T
        slice_EMG_signal_by_tasks(sig1,time_stamps1, EMG_fs)
        slice_EMG_shifted_signal_by_tasks(signal2_after_shift[start_indx_after_shift:,:], time_stamps1, sig2, EMG_fs)

def read_EEG_signal(file_name):
    """
        gets an EDF file with EMG and EEG channels and returns the main EEG channel
        (there are 2: one in the center of the forehead and one on the left)
    :param file_name: path to EMG file
    :return: EEG signal from EMG file
    """
    data = mne.io.read_raw_edf(file_name)
    raw_data = data.get_data()
    EEG_channel = raw_data[15]
    EEG_sig = norm_stand(EEG_channel)
    return EEG_sig

def norm_stand(sig):
    """
        normalizing and standardizing a given signal
    :param signal: signal of a participant
    """
    normalized_sig = (1 / np.max(np.abs(sig))) * sig
    mean = np.mean(normalized_sig)
    std = np.std(normalized_sig, axis=0)
    normalized_standardized_sig = (normalized_sig - mean) / std
    return normalized_standardized_sig

def get_blink_start_time(time_stamps):
    """
        gets a csv file that contains the triggers in EMG file and extracts the starting time of the blinking trigger (from the start of the recording)
    :param time_stamps: a csv file that contains the triggers in EMG file
    :return: starting time of the blinking trigger
    """
    time_stamps_df = pd.read_csv(time_stamps, delimiter=",")
    time_stamps_df = time_stamps_df.set_index('trigger')
    start_blink = time_stamps_df.loc['blink_trig', 'start']
    h_blink, m_blink, s_blink = convert_to_HMS(start_blink, ":")
    start_time_blink = datetime.timedelta(hours=int(h_blink), minutes=int(m_blink), seconds=int(s_blink))
    return start_time_blink

def get_smile_start_time(time_stamps):
    """
        gets a csv file that contains the triggers in EMG file and extracts the starting time of the blinking trigger (from the start of the recording)
    :param time_stamps: a csv file that contains the triggers in EMG file
    :return: starting time of the blinking trigger
    """
    time_stamps_df = pd.read_csv(time_stamps, delimiter=",")
    time_stamps_df = time_stamps_df.set_index('trigger')
    start_smile = time_stamps_df.loc['smile_trig', 'start']
    h_smile, m_smile, s_smile = convert_to_HMS(start_smile, ":")
    start_time_smile = datetime.timedelta(hours=int(h_smile), minutes=int(m_smile), seconds=int(s_smile))
    return start_time_smile

def get_smile_end_time(time_stamps):
    """
        gets a csv file that contains the triggers in EMG file and extracts the starting time of the blinking trigger (from the start of the recording)
    :param time_stamps: a csv file that contains the triggers in EMG file
    :return: starting time of the blinking trigger
    """
    time_stamps_df = pd.read_csv(time_stamps, delimiter=",")
    time_stamps_df = time_stamps_df.set_index('trigger')
    end_smile = time_stamps_df.loc['smile_trig', 'end']
    h_smile, m_smile, s_smile = convert_to_HMS(end_smile, ":")
    end_time_smile = datetime.timedelta(hours=int(h_smile), minutes=int(m_smile), seconds=int(s_smile))
    return end_time_smile

def convert_to_HMS(time_as_string, separator):
    """
        gets a string that represents real time and separates it to hours, minuets, seconds
    :param time_as_string: a string that represents real time. example: "19:35:23"
    :param separator: the char that separates the hours, minuets, seconds. example: ":"
    :return: hours, minuets, seconds
    """
    h = int(time_as_string.split(separator)[0])
    m = int(time_as_string.split(separator)[1])
    s = int(time_as_string.split(separator)[2])
    return h, m, s

def get_blink_end_time(time_stamps):
    """
        gets a csv file that contains the triggers in EMG file and extracts the end time of the blinking trigger (from the start of the recording)
    :param time_stamps: a csv file that contains the triggers in EMG file
    :return: end time of the blinking trigger
    """
    time_stamps_df = pd.read_csv(time_stamps, delimiter=",")
    time_stamps_df = time_stamps_df.set_index('trigger')
    end_blink = time_stamps_df.loc['blink_trig', 'end']
    h_blink, m_blink, s_blink = convert_to_HMS(end_blink, ":")
    end_time_blink = datetime.timedelta(hours=int(h_blink), minutes=int(m_blink), seconds=int(s_blink))
    return end_time_blink

def get_blink_window_from_EMG(sig, time_stamps, fs):
    """
        returns the start and end time stamps of a blinking trigger
    :param sig: EMG signal of a participant
    :param time_stamps: a csv file that contains the triggers time stamps
    :param fs: sampling frequency of the measuring device
    :return: start and end time stamps of a blinking trigger
    """
    start = get_blink_start_time(time_stamps)
    end = get_blink_end_time(time_stamps)
    start_indx = start.seconds * fs
    end_indx = end.seconds * fs
    blink_sig = sig[start_indx:end_indx]
    # plt.figure(figsize=(8,5))
    # plt.plot(blink_sig)
    # plt.rcParams['font.size'] = '16'
    # plt.xlabel("fs*sec", fontsize=16)
    # plt.ylabel("Amplitude", fontsize=16)
    # plt.show()
    # plt.close()
    return start_indx, end_indx

def get_smile_window_from_EMG(sig, time_stamps, fs):
    """
        returns the start and end time stamps of a smiling trigger
    :param sig: EMG signal of a participant
    :param time_stamps: a csv file that contains the triggers time stamps
    :param fs: sampling frequency of the measuring device
    :return: start and end time stamps of a smiling trigger
    """
    start = get_smile_start_time(time_stamps)
    end = get_smile_end_time(time_stamps)
    start_indx = start.seconds * fs
    end_indx = end.seconds * fs
    smile_sig = sig[start_indx:end_indx]
    # plt.figure(figsize=(8,5))
    # plt.plot(smile_sig)
    # plt.rcParams['font.size'] = '16'
    # plt.xlabel("fs*sec", fontsize=16)
    # plt.ylabel("Amplitude", fontsize=16)
    # plt.show()
    # plt.close()
    return start_indx, end_indx

def create_filter(sig, fft_order):
    """
        creates a filter of a pattern of a participant using FFT in order to recognize it in a signal
    :param sig: EMG signal of a participant
    :param fft_order: the requested number of Fourier coefficients to calculate for the filter
    :param time_stamps: a csv file that contains the triggers time stamps
    :return: a filter of a pattern
    """
    n = len(sig)
    fftsig = np.fft.fft(sig)
    np.put(fftsig, range(fft_order + 1, n), 0.0)
    win = np.fft.ifft(fftsig)
    win1 = np.real(np.fft.ifft(fftsig)) / np.sqrt(np.sum(np.square(win)))
    # plt.figure(figsize=(9,5))
    # plt.rcParams['font.size'] = '16'
    # plt.xlabel("fs*sec", fontsize=16)
    # plt.ylabel("Amplitude", fontsize=16)
    # plt.plot(win1)
    # plt.show()
    return win1

def convert_patten_to_spike_train(sig, filter, fs):
    """
        gets an EMG signal and a blinking pattern filter and converts the signal to a spike train of blinks
    :param sig: EMG signal of a participant
    :param filter: a blinking pattern filter
    :param fs: sampling frequency of the measuring device
    :param time_stamps: a csv file that contains the triggers time stamps
    :return: a spike train of blinks
    """
    N = len(sig)
    t = np.linspace(1 / fs, N, N)
    corr = signal.correlate(sig, filter, mode='same') / len(filter)
    peaks = signal.find_peaks(corr, height=0.01)
    spike_train = np.zeros_like(t)
    spike_train[peaks[0]] = 1
    # plt.figure(figsize=(8, 5))
    # plt.rcParams['font.size'] = '16'
    # plt.plot(t, np.real(corr))
    # plt.plot(peaks[0], peaks[1]['peak_heights'], '*')
    # plt.xlabel("fs*sec", fontsize=16)
    # plt.ylabel("cross-correlation", fontsize=16)
    # plt.show()
    # plt.close()
    # plt.show()
    # plt.close()
    # plt.figure()
    # plt.rcParams['font.size'] = '16'
    # plt.xlabel("fs*sec", fontsize=14)
    # plt.ylabel("occurrence of a spike", fontsize=14)
    # plt.stem(t, spike_train)
    # plt.show()
    # plt.close()
    return spike_train

def calc_RMS(spike_train1, spike_train2, fs):
    """
        calculates the root mean squares (RMS) of two spike trains. used as a synchronization analysis.
    :param spike_train1: spike train of participant1
    :param spike_train2: spike train of participant2
    :param fs: sampling frequency of the measuring device
    :return: RMS of two spike trains
    """
    h1, bins1, hist1, h2, bins2, hist2 = calc_CIH(spike_train1,spike_train2, fs)
    RMS1 = np.sqrt(np.mean(h1 ** 2)) / fs
    RMS2 = np.sqrt(np.mean(h2 ** 2)) / fs
    return min(RMS1, RMS2)

def calc_CIH(spike_train1, spike_train2, fs):
    """
        calculates the cross-interval histogram (CIH) of two spike trains
    :param spike_train1: spike train of participant1
    :param spike_train2: spike train of participant2
    :param fs: sampling frequency of the measuring device
    :return: CIH of two spike trains
    """
    blinks_num1 = np.where(spike_train1 == 1)[0]
    blinks_num2 = np.where(spike_train2 == 1)[0]
    diff_mat = np.zeros((len(blinks_num1), len(blinks_num2)))
    for i in range(len(blinks_num1)):
        diff_mat[i] = np.abs(blinks_num2 - blinks_num1[i])
    h1 = np.min(diff_mat, axis=0).transpose()
    h2 = np.min(diff_mat, axis=1)
    tau1 = h1 / fs
    tau2 = h2 / fs
    maximum_distance = max(np.max(tau1), np.max(tau2))
    hist_bins = np.linspace(-0.5 / fs, maximum_distance + 0.5 / fs, int(maximum_distance * fs + 1))
    hist1 = np.histogram(tau1, hist_bins)
    hist2 = np.histogram(tau2, hist_bins)
    return h1, hist_bins[:-1], hist1[0], h2, hist_bins[:-1], hist2[0]

def slice_EMG_signal_by_tasks(EMG_signal, time_stamps, fs):
    """
        receives an EMG signal and returns a csv file describing the signal in each task
    :param EMG_signal: EMG signal as an edf file
    :param time_stamps: a csv file that contains the triggers time stamps
    """
    data = mne.io.read_raw_edf(EMG_signal)
    time_stamps_df = pd.read_csv(time_stamps, delimiter=",")
    time_stamps_df = time_stamps_df.set_index('trigger')
    last_end_ind = 0
    last_trig_name = ''
    flag = False
    for row in time_stamps_df.iterrows():
        if  last_end_ind == 0:
            timedelt_start_time, timedelt_end_time, trig_name = extract_time_stamps(row)
            start_ind = timedelt_start_time.seconds * fs
            end_ind = timedelt_end_time.seconds * fs
            last_end_ind = end_ind
            last_trig_name = trig_name
            signal_in_task = (data.get_data()[0:16, start_ind:end_ind]).T
            EMG_in_task_df = pd.DataFrame(signal_in_task)
            EMG_in_task_df.to_csv(fr"{os.path.dirname(EMG_signal)}/signal_in_{trig_name}.csv")
        else:
            timedelt_start_time, timedelt_end_time, trig_name = extract_time_stamps(row)
            if last_trig_name == 'audio1':
                flag = True
            start_ind = timedelt_start_time.seconds * fs
            end_ind = timedelt_end_time.seconds * fs
            signal_in_task = (data.get_data()[0:16, start_ind:end_ind]).T
            EMG_in_task_df = pd.DataFrame(signal_in_task)
            EMG_in_task_df.to_csv(fr"{os.path.dirname(EMG_signal)}/signal_in_{trig_name}.csv")
            if flag :
                signal_between_tasks = (data.get_data()[0:16, last_end_ind:start_ind]).T
                EMG_between_task_df = pd.DataFrame(signal_between_tasks)
                EMG_between_task_df.to_csv(fr"{os.path.dirname(EMG_signal)}/signal_between_{last_trig_name} and {trig_name}.csv")
            last_end_ind = end_ind
            last_trig_name = trig_name

def extract_time_stamps(row):
    """
        extracts time stamps of a given trigger from a csv file of time stamps of all the triggers
    :param row: a row from the csv file of the time stamps of all the triggers
    :return: start, end time stamps as timedelta object and the triggers name
    """
    trig_name = row[0]
    trig_start_time = row[1][0]
    trig_end_time = row[1][1]
    h_start, m_start, s_start = convert_to_HMS(trig_start_time, ":")
    h_end, m_end, s_end = convert_to_HMS(trig_end_time, ":")
    timedelt_start_time = datetime.timedelta(hours=int(h_start), minutes=int(m_start), seconds=int(s_start))
    timedelt_end_time = datetime.timedelta(hours=int(h_end), minutes=int(m_end), seconds=int(s_end))
    return timedelt_start_time, timedelt_end_time, trig_name

def slice_EMG_shifted_signal_by_tasks(EMG_signal, time_stamps, path, fs):
    """
        receives a shifted EMG signal and returns a csv file describing the signal in each task
    :param EMG_signal: EMG signal as a ndarray
    :param time_stamps: a csv file that contains the triggers in EMG file
    """
    time_stamps_df = pd.read_csv(time_stamps, delimiter=",")
    time_stamps_df = time_stamps_df.set_index('trigger')
    last_end_ind = 0
    last_trig_name = ''
    flag = False
    for row in time_stamps_df.iterrows():
        if last_end_ind == 0:
            timedelt_start_time, timedelt_end_time, trig_name = extract_time_stamps(row)
            start_ind = timedelt_start_time.seconds * fs
            end_ind = timedelt_end_time.seconds * fs
            last_end_ind = end_ind
            last_trig_name = trig_name
            signal_in_task = EMG_signal[start_ind:end_ind, :]
            EMG_in_task_df = pd.DataFrame(signal_in_task)
            EMG_in_task_df.to_csv(fr"{os.path.dirname(path)}/signal_in_{trig_name}.csv")
        else:
            timedelt_start_time, timedelt_end_time, trig_name = extract_time_stamps(row)
            if last_trig_name == 'audio1':
                flag = True
            start_ind = timedelt_start_time.seconds * fs
            end_ind = timedelt_end_time.seconds * fs
            signal_in_task = EMG_signal[start_ind:end_ind, :]
            signal_between_tasks = EMG_signal[last_end_ind:start_ind, :]
            EMG_in_task_df = pd.DataFrame(signal_in_task)
            EMG_in_task_df.to_csv(fr"{os.path.dirname(path)}/signal_in_{trig_name}.csv")
            if flag:
                EMG_between_task_df = pd.DataFrame(signal_between_tasks)
                EMG_between_task_df.to_csv(fr"{os.path.dirname(path)}/signal_between_{last_trig_name} and {trig_name}.csv")
            last_end_ind = end_ind
            last_trig_name = trig_name

def align_breath(breath_signal1, breath_signal2, time_stamps1, time_stamps2):
    breath_signal1_df = pd.read_csv(breath_signal1, delimiter=",")
    breath_signal2_df = pd.read_csv(breath_signal2, delimiter=",")
    breath_signal1_df.loc[:, 'pressure2+'] = -1 * (breath_signal1_df.loc[:, 'pressure2'])
    breath_signal2_df.loc[:, 'pressure2+'] = -1 * (breath_signal2_df.loc[:, 'pressure2'])
    signal1 = (breath_signal1_df.loc[:, ['pressure1', 'pressure2+']])
    signal2 = (breath_signal2_df.loc[:, ['pressure1', 'pressure2+']])
    breath_fs = 20
    slice_breath_signal_by_tasks(signal1, breath_signal1, time_stamps1, breath_fs)
    slice_breath_signal_by_tasks(signal2, breath_signal2, time_stamps2, breath_fs)


def slice_breath_signal_by_tasks(breath_signal, path, time_stamps, fs):
    """
         receives a breath signal and returns a csv file describing the signal in each task
    :param breath_signal: breath signal as a data frame
    :param time_stamps: a csv file that contains the triggers in EMG file
    """
    time_stamps_df = pd.read_csv(time_stamps, delimiter=",")
    time_stamps_df = time_stamps_df.set_index('trigger')
    last_end_ind = 0
    last_trig_name = ''
    flag = False
    for row in time_stamps_df.iterrows():
        if last_end_ind == 0:
            timedelt_start_time, timedelt_end_time, trig_name = extract_time_stamps_breath(row)
            start_ind = timedelt_start_time.seconds * fs
            end_ind = timedelt_end_time.seconds * fs
            last_end_ind = end_ind
            last_trig_name = trig_name
            signal_in_task = breath_signal.loc[start_ind:end_ind,:]
            signal_in_task.to_csv(fr"{os.path.dirname(path)}/signal_in_{trig_name}.csv")
        else:
            timedelt_start_time, timedelt_end_time, trig_name = extract_time_stamps_breath(row)
            if last_trig_name == 'audio1':
                flag = True
            start_ind = timedelt_start_time.seconds * fs
            end_ind = timedelt_end_time.seconds * fs
            signal_in_task = breath_signal.loc[start_ind:end_ind,:]
            signal_in_task.to_csv(fr"{os.path.dirname(path)}/signal_in_{trig_name}.csv")
            if flag:
                signal_between_tasks = breath_signal.loc[last_end_ind:start_ind,:]
                signal_between_tasks.to_csv(
                    fr"{os.path.dirname(path)}/signal_between_{last_trig_name} and {trig_name}.csv")
            last_end_ind = end_ind
            last_trig_name = trig_name

def extract_time_stamps_breath(row):
    """
        extracts time stamps of a given trigger from a csv file of time stamps of all the triggers
    :param row: a row from the csv file of the time stamps of all the triggers
    :return: start, end time stamps as timedelta object and the triggers name
    """
    trig_name = row[0]
    trig_start_time = row[1][1]
    trig_end_time = row[1][2]
    timedelt_start_time = datetime.timedelta(seconds=int(trig_start_time))
    timedelt_end_time = datetime.timedelta(seconds=int(trig_end_time))
    return timedelt_start_time, timedelt_end_time, trig_name

def get_respiration_window(sig, time_stamps, fs):
    start = get_breath_start_time(time_stamps)
    end = get_breath_end_time(time_stamps)
    start_indx = start.seconds * fs
    end_indx = end.seconds * fs
    breath_sig = sig[start_indx:end_indx]
    # breath_sig = sig[start_indx - fs * 10: end_indx + fs * 10]
    # plt.figure()
    # plt.plot(breath_sig)
    # plt.title(fr"breath trigger {os.path.basename(time_stamps).split('_')[-1].split('.')[0]}")
    # plt.show()
    # plt.close()
    return start_indx, end_indx

def get_breath_start_time(time_stamps):
    """
        gets a csv file that contains the triggers in EMG file and extracts the starting time of the blinking trigger (from the start of the recording)
    :param time_stamps: a csv file that contains the triggers in EMG file
    :return: starting time of the first trigger
    """
    time_stamps_df = pd.read_csv(time_stamps, delimiter=",")
    time_stamps_df = time_stamps_df.set_index('trigger')
    start_breath = time_stamps_df.loc['breath_trig', 'start']
    start_breath_delt = datetime.timedelta(seconds=int(start_breath))
    return start_breath_delt

def get_breath_end_time(time_stamps):
    """
        gets a csv file that contains the triggers in EMG file and extracts the starting time of the blinking trigger (from the start of the recording)
    :param time_stamps: a csv file that contains the triggers in EMG file
    :return: starting time of the first trigger
    """
    time_stamps_df = pd.read_csv(time_stamps, delimiter=",")
    time_stamps_df = time_stamps_df.set_index('trigger')
    end_breath = time_stamps_df.loc['breath_trig', 'end']
    end_breath_delt = datetime.timedelta(seconds=int(end_breath))
    return end_breath_delt

def get_blink_real_time(time_stamps):
    """
        gets a csv file that contains the triggers in EMG file and extracts the starting time of the first trigger
    :param time_stamps: a csv file that contains the triggers in EMG file
    :return: starting time of the first trigger
    """
    time_stamps_df = pd.read_csv(time_stamps, delimiter=",")
    time_stamps_df = time_stamps_df.set_index('trigger')
    start_blink = time_stamps_df.loc['blink_trig', 'real time start']
    h_blink, m_blink, s_blink = convert_to_HMS(start_blink, ":")
    real_start_time_blink = datetime.timedelta(hours=int(h_blink), minutes=int(m_blink), seconds=int(s_blink))
    return real_start_time_blink
#
def shift_breath(breath_signal_describe, breath_signal, time_stamps):
    """
        gets a breathing signal and shifts it in time according to a trigger in EMG signal (given in a csv file).
        Extracts a csv file of the shifted signal.
    :param breath_signal_describe: a txt file that describes the starting time of the breathing signal
    :param breath_signal: breathing signal given as a csv file
    :param time_stamps: a csv file that contains the triggers in EMG file
    """
    with open(breath_signal_describe, 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
    time_stamp_breath = ((last_line.split(" ")[1]).split(",")[0]).split(".")[0]
    h_breath, m_breath, s_breath = convert_to_HMS(time_stamp_breath, ":")
    real_start_time_breath = datetime.timedelta(hours=int(h_breath), minutes=int(m_breath), seconds=int(s_breath))

    real_start_time_blink = get_blink_real_time(time_stamps)
    if not (os.path.exists(fr'{os.path.dirname(time_stamps)}/breath time stamps.csv')):
        create_breath_time_stamps(time_stamps, real_start_time_blink - real_start_time_breath)

    breath_signal_df = pd.read_csv(breath_signal, delimiter=",")
    breath_df = breath_signal_df.loc[:, 'time']
    breath_df = pd.DataFrame(pd.to_timedelta(breath_df, 'seconds'))
    time_shift = breath_df[breath_df <= (real_start_time_blink - real_start_time_breath)].dropna()
    breath_after_shift = breath_signal_df[len(time_shift):]
    breath_after_shift.to_csv(fr'{os.path.dirname(breath_signal)}/breath_after_shift.csv')

def create_breath_time_stamps(time_stamps, real_start_time_breath):
    time_stamps_df = pd.read_csv(time_stamps, delimiter=",")
    time_stamps_df = time_stamps_df.set_index('trigger')
    start_time_blink = get_blink_start_time(time_stamps)
    breath_stamps_df = pd.DataFrame({'trigger':[], 'start':[], 'end':[]})
    for row in time_stamps_df.iterrows():
        timedelt_start_time, timedelt_end_time, trig_name = extract_time_stamps(row)
        new_row = pd.DataFrame({'trigger': row[0], 'start': [
            timedelt_start_time.seconds - start_time_blink.seconds + real_start_time_breath.seconds],
                            'end': [timedelt_end_time.seconds - start_time_blink.seconds + real_start_time_breath.seconds]})
        breath_stamps_df = pd.concat([breath_stamps_df, new_row])
    breath_stamps_df.to_csv(fr'{os.path.dirname(time_stamps)}/breath time stamps.csv')

# def shift_fNIRS(fNIRS_signal_describe, fNIRS_signal_file_name, time_stamps):
#     """
#         Gets a fNIRS signal and shifts it in time according to a trigger in EMG signal (given in a csv file).
#         Extracts an edf file of the shifted signal.
#     :param fNIRS_signal_describe: a path to file with name that describes starting time of the fNIRS signal
#     :param fNIRS_signal_file_name: fNIRS signal given as an edf file
#     :param time_stamps: a csv file that contains the triggers in EMG file
#     """
#     temp_shift = (os.path.basename(fNIRS_signal_describe).split("-")[3]).split(".")[0]
#     h_fNIRS, m_fNIRS, s_fNIRS = convert_to_HMS(temp_shift, "_")
#     real_start_time_fNIRS = datetime.timedelta(hours=int(h_fNIRS), minutes=int(m_fNIRS), seconds=int(s_fNIRS))
#     real_start_time_blink = get_blink_real_time(time_stamps)
#     data = mne.io.read_raw_edf(fNIRS_signal_file_name)
#     fs = 50
#     shift = (real_start_time_blink - real_start_time_fNIRS).seconds * fs
#     fnirs_signal_after_shift = data.get_data()[:, shift:]
#     fnirs_after_shift_df = pd.DataFrame(fnirs_signal_after_shift.T)
#     # fnirs_after_shift_df.to_csv(fr"{os.path.dirname(fNIRS_signal_file_name)}/fNIRS_after_shift.csv")
#
# def slice_fNIRS_signal_by_tasks(fNIRS_signal_after_shift, time_stamps):
#     """
#         receives a fNIRS signal and returns a csv file describing the signal in each task
#     :param fNIRS_signal_after_shift: fNIRS signal as a csv file
#     :param time_stamps: a csv file that contains the triggers in EMG file
#     """
#     fNIRS_sig = pd.read_csv(fNIRS_signal_after_shift, delimiter=",")
#     fs = 50
#     slice_signal(fNIRS_sig, fNIRS_signal_after_shift, fs, time_stamps)
#
# def slice_signal(signal, signal_after_shift, fs, time_stamps):
#     time_stamps_df = pd.read_csv(time_stamps, delimiter=",")
#     time_stamps_df = time_stamps_df.set_index('trigger')
#     start_time_blink = get_blink_start_time(time_stamps)
#     for row in time_stamps_df.loc['audio1':'audio2', :].iterrows():
#         timedelt_start_time, timedelt_end_time, trig_name = extract_time_stamps(row)
#         start_ind = (timedelt_start_time.seconds - start_time_blink.seconds) * fs
#         end_ind = (timedelt_end_time.seconds - start_time_blink.seconds) * fs
#         signal_in_task = signal.iloc[start_ind:end_ind, :]
#         signal_in_task.to_csv(fr"{os.path.dirname(signal_after_shift)}/signal_in_{trig_name}.csv")
#
# def slice_breath_signal_by_tasks(breath_signal_after_shift, time_stamps):
#     """
#          receives a breath signal and returns a csv file describing the signal in each task
#     :param breath_signal_after_shift: breath signal as a csv file
#     :param time_stamps: a csv file that contains the triggers in EMG file
#     """
#     breath_sig = pd.read_csv(breath_signal_after_shift, delimiter=",")
#     fs = 20
#     slice_signal(breath_sig, breath_signal_after_shift, fs, time_stamps)
#
# def shift_HR(HR_signal_file_name, time_stamps):
#     """
#         Gets a heart beat signal and shifts it in time according to a trigger in EMG signal (given in a csv file).
#         Extracts a csv file of the shifted signal.
#     :param HR_signal_file_name: heart beat signal given as a txt file (beat intervals)
#     :param time_stamps: a csv file that contains the triggers in EMG file
#     """
#     temp_df = pd.read_csv(HR_signal_file_name, header=None)
#     HR_signal = temp_df.to_numpy()
#     time_diffs = np.cumsum(HR_signal)
#     time_diffs = pd.DataFrame(pd.to_timedelta(time_diffs, 'milliseconds'))
#     temp_shift = (os.path.basename(HR_signal_file_name).split(" ")[1]).split(".")[0]
#     h_HR, m_HR, s_HR = convert_to_HMS(temp_shift, "-")
#     real_start_time_HR = datetime.timedelta(hours=h_HR, minutes=m_HR, seconds=s_HR)
#     real_start_time_blink = get_blink_real_time(time_stamps)
#     time_shift = time_diffs[time_diffs <= (real_start_time_blink - real_start_time_HR)].dropna()
#     HR_after_shift = HR_signal[len(time_shift):]
#     # pd.DataFrame(HR_after_shift).to_csv(fr"{os.path.dirname(HR_signal_file_name)}/HR_after_shift.csv")
#
# def slice_HR_signal_by_tasks(HR_signal_after_shift, time_stamps):
#     """
#         receives a HR signal and returns a csv file describing the signal in each task
#     :param HR_signal_after_shift: Heart Rate signal as a csv file
#     :param time_stamps: a csv file that contains the triggers in EMG file
#     """
#     temp_df = pd.read_csv(HR_signal_after_shift, delimiter=",")
#     HR_signal = temp_df.iloc[:, 1].to_numpy()
#     time_diffs = np.cumsum(HR_signal)
#     time_diffs = pd.DataFrame(pd.to_timedelta(time_diffs, 'milliseconds'))
#     time_stamps_df = pd.read_csv(time_stamps, delimiter=",")
#     time_stamps_df = time_stamps_df.set_index('trigger')
#     start_time_blink = get_blink_start_time(time_stamps)
#     for row in time_stamps_df.loc['audio1':'audio2', :].iterrows():
#         timedelt_start_time, timedelt_end_time, trig_name = extract_time_stamps(row)
#         start_ind = time_diffs[time_diffs <= (timedelt_start_time - start_time_blink)].dropna()
#         end_ind = time_diffs[time_diffs <= (timedelt_end_time - start_time_blink)].dropna()
#         signal_in_task = HR_signal[len(start_ind):len(end_ind)]
#         # pd.DataFrame(signal_in_task).to_csv(fr"{os.path.dirname(HR_signal_after_shift)}/signal_in_{trig_name}.csv")