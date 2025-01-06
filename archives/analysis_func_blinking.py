import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats
from scipy import signal
import math
import preprocess_func_blinking as preprocess_func

def EMG_describe_std_mean_all_tasks_xcorr(tasks_lst1, tasks_lst2):
    """
        calculates the STD and mean of the correlation of the signals of
        two participants during each task and between tasks
    :param tasks_lst1: a list of csv paths that contain the signal of participant1
    :param tasks_lst2: a list of csv paths that contain the signal of participant2
    """
    mean_std_per_task = np.zeros((len(tasks_lst1), 17, 2))
    # print("Correlation statistics")
    for i in range(len(tasks_lst1)):
        signal1, signal2 = preproccess_signals(tasks_lst1[i], tasks_lst2[i], 'norm')
        ac_mean, ac_std = get_mean_std_of_xcorr(signal1, signal2, 4000)
        # print(f"{tasks_lst1[i]} - mean = {ac_mean}, std = {ac_std}")
        for j in range(np.shape(signal1)[1]):
            mean_std_per_task[i, j, 0] = ac_mean[j]
            mean_std_per_task[i, j, 1] = ac_std[j]
    for i in range(17):
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.bar(range(12), mean_std_per_task[:, i, 0], yerr=mean_std_per_task[:, i, 1], align='center', alpha=0.5,
               ecolor='black', capsize=10)
        ax.set_ylabel('Cross-Correlation', fontsize=20)
        ax.set_xlabel('tasks', fontsize=20)
        plt.rcParams['font.size'] = '22'
        ax.set_xticks(range(12))
        ax.set_xticklabels(['blink trig', 'audio1', 'between', 'blink', 'between', 'button',
                            'between', 'mirror', 'between', 'smile', 'between', 'audio2'], fontsize=16)
        fig.show()
        fig.savefig(fr'Cross-Correlation mean and std of each task in channel {i}')
    plt.close()

def breath_describe_std_mean_all_tasks_xcorr(tasks_lst1, tasks_lst2):
    """
        calculates the STD and mean of the correlation of the signals of
        two participants during each task and between tasks
    :param tasks_lst1: a list of csv paths that contain the signal of participant1
    :param tasks_lst2: a list of csv paths that contain the signal of participant2
    """
    mean_std_per_task = np.zeros((len(tasks_lst1), 3, 2))
    # print("Correlation statistics")
    for i in range(len(tasks_lst1)):
        signal1, signal2 = breath_preproccess_signals(tasks_lst1[i], tasks_lst2[i], 'norm')
        ac_mean, ac_std = get_mean_std_of_xcorr(signal1, signal2, 20, trig_time = 7, overlap=3)
        # print(f"{tasks_lst1[i]} - mean = {ac_mean}, std = {ac_std}")
        for j in range(np.shape(signal1)[1]):
            mean_std_per_task[i, j, 0] = ac_mean[j]
            mean_std_per_task[i, j, 1] = ac_std[j]
    for i in range(3):
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.bar(range(12), mean_std_per_task[:, i, 0], yerr=mean_std_per_task[:, i, 1], align='center', alpha=0.5,
               ecolor='black', capsize=10)
        ax.set_ylabel('Cross-Correlation', fontsize=20)
        ax.set_xlabel('tasks', fontsize=20)
        plt.rcParams['font.size'] = '22'
        ax.set_xticks(range(12))
        ax.set_xticklabels(['trig', 'audio1', 'between', 'blink', 'between', 'button',
                            'between', 'mirror', 'between', 'smile', 'between', 'audio2'], fontsize=16)
        fig.show()
        plt.savefig(fr'breath Correlation mean and std of each task in channel {i}')
    plt.close()
def preproccess_signals(EMG_signal1, EMG_signal2, flag):
    """
        reads EMG signals from csv files, applies notch and band pass filter and normalized the signals
    :param EMG_signal1: path to EMG file of participant1
    :param EMG_signal2: path to EMG file of participant2
    :param flag: a flag that indicates on the wanted calculation - normalize and standardize or just normalize the signals
    :return: two pre-processed signals
    """
    signal1_df = pd.read_csv(EMG_signal1, delimiter=",")
    signal2_df = pd.read_csv(EMG_signal2, delimiter=",")
    signal1 = signal1_df.to_numpy()
    signal2 = signal2_df.to_numpy()

    fs = 4000
    notch_freq = 50.0
    quality_factor = 30.0
    signal1 = notch_filter(notch_freq, quality_factor, signal1, fs)
    signal2 = notch_filter(notch_freq, quality_factor, signal2, fs)


    signal1 = band_pass_filter(signal1, fs)
    signal2 = band_pass_filter(signal2, fs)
    if flag == 'norm':
        signal1 = normalize_signal(signal1)
        signal2 = normalize_signal(signal2)
    else:
        signal1 = preprocess_func.norm_stand(signal1)
        signal2 = preprocess_func.norm_stand(signal2)
    return signal1, signal2

def breath_preproccess_signals(breath_signal1, breath_signal2, flag):
    """
        reads EMG signals from csv files, applies notch and band pass filter and normalized the signals
    :param breath_signal1: path to EMG file of participant1
    :param breath_signal2: path to EMG file of participant2
    :param flag: a flag that indicates on the wanted calculation - normalize and standardize or just normalize the signals
    :return: two pre-processed signals
    """
    signal1_df = pd.read_csv(breath_signal1, delimiter=",")
    signal2_df = pd.read_csv(breath_signal2, delimiter=",")
    signal1 = signal1_df.to_numpy()
    signal2 = signal2_df.to_numpy()
    if flag == 'norm':
        signal1 = normalize_signal(signal1)
        signal2 = normalize_signal(signal2)
    else:
        signal1 = preprocess_func.norm_stand(signal1)
        signal2 = preprocess_func.norm_stand(signal2)
    return signal1, signal2

def notch_filter(notch_freq, quality_factor, sig, fs):
    """
       applying notch filter to EMG and EEG signals to relevant frequencies to a signal of a participant
       :param sig: signal of a participant
       :param fs: sampling frequency of the measuring device
       :param quality_factor: The quality factor shows how narrow or wide the stop band is for a notch filter.
       :return: notch filtered signal
       """
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, fs)
    for i in range(16):
        filtered = signal.filtfilt(b_notch, a_notch, sig.T[i])
        sig.T[i] = filtered
    return sig

def band_pass_filter(sig, fs):
    """
        applying band pass filter to EMG and EEG signals to relevant frequencies to a signal of a participant
    :param sig: signal of a participant
    :param fs: sampling frequency of the measuring device
    :return: band pass filtered signal
    """
    sos = signal.butter(4, [35, 350], 'bp', fs=fs, output='sos')
    for i in range(14):
        filtered_EMG = signal.sosfilt(sos, sig.T[i])
        sig.T[i] = filtered_EMG
    sos = signal.butter(4, [1, 35], 'bp', fs=fs, output='sos')
    for i in [14,15]:
        filtered_EEG = signal.sosfilt(sos, sig.T[i])
        sig.T[i] = filtered_EEG
    return sig

def normalize_signal(sig):
    """
        normalizing a given signal
    :param sig: a signal of a participant
    """
    return sig/np.sqrt(np.sum(np.power(sig,2), axis=0))

def get_mean_std_of_xcorr(signal1, signal2, fs, trig_time=7, overlap=4):
    """
        calculates cross-correlation between two signals in each window and returns the mean and STD of all the windows
    :param signal1: EMG signal of participant1
    :param signal2: EMG signal of participant2
    :param fs: sampling frequency of the measuring device
    :param trig_time: length of the blinking trigger, the maximum lag for calculating cross-correlation
    :param overlap: overlap between each two windows
    :return: mean and STD of cross-correlation in all the windows
    """
    t = trig_time / overlap
    all_sig_corr = np.zeros((math.floor(len(signal1) / (fs * t)) - overlap + 1, np.shape(signal1)[1]))
    for i in range(math.floor(len(signal1) / (fs * t))-overlap+1):
        for j in range(np.shape(signal1)[1]):
            all_sig_corr[i, j] = np.max(xcorr(signal1[math.floor(i * fs * t):math.floor((i + overlap) * fs * t), j],
                                              signal2[math.floor(i * fs * t):math.floor((i + overlap) * fs * t), j],
                                              max_lag=np.round(fs), mode='full')[1])
    ac_mean = all_sig_corr.mean(axis=0)
    ac_std = all_sig_corr.std(axis=0)
    return ac_mean, ac_std

def xcorr(signal1, signal2, max_lag=0, scaleopt='none', mode='full'):
    """
    Calculate cross-correlation between two signals up to a maximal lag.
    If max_lag is 0, then include all lags.
    scaleopt can be 'none' (for bare cross-correlation) or 'biased' (which normalized by the signal length).
    :param signal1: EMG signal of participant1
    :param signal2: EMG signal of participant2
    :return: lags and correlation value arrays for each lag
    """
    corr = signal.correlate(signal1, signal2, mode=mode)
    # lags = signal.correlation_lags(len(x), len(y), mode="full")
    if mode == 'same':
        lags = np.arange(-2, len(signal1))
    else:
        lags = np.arange(-len(signal1) + 1, len(signal1))
    if scaleopt == 'biased':
        corr = corr / len(signal1)
    if max_lag > 0:
        indices = np.where(np.abs(lags) <= max_lag)[0]
        corr = corr[indices]
        lags = lags[indices]
    return lags, corr

def describe_std_mean_all_tasks_coherence(tasks_lst1, tasks_lst2):
    """
        calculates the STD and mean of the coherence of the signals of
        two participants during each task and between tasks
        :param tasks_lst1: a list of csv paths that contain the signal of participant1
        :param tasks_lst2: a list of csv paths that contain the signal of participant2
        """
    fs = 4000
    mean_std_per_task = np.zeros((len(tasks_lst1), 17, 2))
    # print("Coherence statistics")
    for i in range(len(tasks_lst1)):
        signal1, signal2 = preproccess_signals(tasks_lst1[i], tasks_lst2[i], 'norm')
        ac_mean, ac_std = get_mean_std_of_coherence(signal1, signal2, fs)
        # print(f"{tasks_lst1[i]} - mean = {ac_mean}, std = {ac_std}")
        for j in range(np.shape(signal1)[1]):
            mean_std_per_task[i, j, 0] = ac_mean[j]
            mean_std_per_task[i, j, 1] = ac_std[j]
    for i in [15,16]:
        fig, ax = plt.subplots(figsize=(16,6))
        ax.bar(range(12), mean_std_per_task[:, i, 0], yerr=mean_std_per_task[:, i, 1], align='center', alpha=0.5,
               ecolor='black', capsize=10)
        ax.set_ylabel('Coherence', fontsize=20)
        ax.set_xlabel('tasks', fontsize=20)
        plt.rcParams['font.size'] = '18'
        ax.set_xticks(range(12))
        ax.set_xticklabels(['trig', 'audio1', 'between', 'blink', 'between', 'button',
                            'between', 'mirror', 'between', 'smile', 'between', 'audio2'], fontsize=16)
        fig.show()
        fig.savefig(fr'Coherence mean and std of each task in channel {i}')
    for i in [5]:
        fig, ax = plt.subplots(figsize=(16,6))
        ax.bar(range(12), mean_std_per_task[:, i, 0], yerr=mean_std_per_task[:, i, 1], align='center', alpha=0.5,
               ecolor='black', capsize=10)
        ax.set_ylabel('Coherence', fontsize=20)
        ax.set_xlabel('tasks', fontsize=20)
        plt.rcParams['font.size'] = '20'
        ax.set_xticks(range(12))
        ax.set_xticklabels(['trig', 'audio1', 'between', 'blink', 'between', 'button',
                            'between', 'mirror', 'between', 'smile', 'between', 'audio2'], fontsize=16)
        fig.show()
        fig.savefig(fr'Coherence mean and std of each task in channel {i}')

    plt.close()

def breath_describe_std_mean_all_tasks_coherence(tasks_lst1, tasks_lst2):
    """
        calculates the STD and mean of the coherence of the signals of
        two participants during each task and between tasks
        :param tasks_lst1: a list of csv paths that contain the signal of participant1
        :param tasks_lst2: a list of csv paths that contain the signal of participant2
        """
    fs = 20
    mean_std_per_task = np.zeros((len(tasks_lst1), 3, 2))
    # print("Coherence statistics")
    for i in range(len(tasks_lst1)):
        signal1, signal2 = breath_preproccess_signals(tasks_lst1[i], tasks_lst2[i], 'norm')
        ac_mean, ac_std = breath_get_mean_std_of_coherence(signal1, signal2, fs)
        # print(f"{tasks_lst1[i]} - mean = {ac_mean}, std = {ac_std}")
        for j in range(np.shape(signal1)[1]):
            mean_std_per_task[i, j, 0] = ac_mean[j]
            mean_std_per_task[i, j, 1] = ac_std[j]
    for i in range(3):
        fig, ax = plt.subplots(figsize=(16,6))
        ax.bar(range(12), mean_std_per_task[:, i, 0], yerr=mean_std_per_task[:, i, 1], align='center', alpha=0.5,
               ecolor='black', capsize=10)
        ax.set_ylabel('Coherence', fontsize=20)
        ax.set_xlabel('tasks', fontsize=20)
        plt.rcParams['font.size'] = '18'
        ax.set_xticks(range(12))
        ax.set_xticklabels(['trig', 'audio1', 'between', 'blink', 'between', 'button',
                            'between', 'mirror', 'between', 'smile', 'between', 'audio2'], fontsize=16)
        fig.show()
        fig.savefig(fr'breath Coherence mean and std of each task in channel {i}')

    plt.close()

def breath_get_mean_std_of_coherence(signal1, signal2, fs, blink_trig_time=7, overlap=4, ):
    """
        calculates coherence between two signals in each window and returns the mean and STD of all the windows
        :param signal1: EMG signal of participant1
        :param signal2: EMG signal of participant2
        :param fs: sampling frequency of the measuring device
        :param blink_trig_time: length of the blinking trigger and window to calculate
        :param overlap: overlap between each two windows
        :return: mean and STD of coherence in all the windows
        """
    start = 0
    end = 2
    t = blink_trig_time / overlap
    all_sig_corr = np.zeros((math.floor(len(signal1) / (fs * t)) - overlap + 1, np.shape(signal1)[1]))
    for i in range(math.floor(len(signal1) / (fs * t))-overlap+1):
        for j in range(np.shape(signal1)[1]):

            all_sig_corr[i, j] = np.max(coherence(signal1[math.floor(i * fs * t):math.floor((i + overlap) * fs * t), j],
                                              signal2[math.floor(i * fs * t):math.floor((i + overlap) * fs * t), j],
                                              fs)[1][start:end])
    ac_mean = all_sig_corr.mean(axis=0)
    ac_std = all_sig_corr.std(axis=0)
    return ac_mean, ac_std


def get_mean_std_of_coherence(signal1, signal2, fs, blink_trig_time=7, overlap=4, ):
    """
        calculates coherence between two signals in each window and returns the mean and STD of all the windows
        :param signal1: EMG signal of participant1
        :param signal2: EMG signal of participant2
        :param fs: sampling frequency of the measuring device
        :param blink_trig_time: length of the blinking trigger and window to calculate
        :param overlap: overlap between each two windows
        :return: mean and STD of coherence in all the windows
        """
    t = blink_trig_time / overlap
    all_sig_corr = np.zeros((math.floor(len(signal1) / (fs * t)) - overlap + 1, np.shape(signal1)[1]))
    for i in range(math.floor(len(signal1) / (fs * t))-overlap+1):
        for j in range(np.shape(signal1)[1]):
            if j in [5]:
                start = 0
                end = 20
            else:
                start = 0
                end = 40
            all_sig_corr[i, j] = np.max(coherence(signal1[math.floor(i * fs * t):math.floor((i + overlap) * fs * t), j],
                                              signal2[math.floor(i * fs * t):math.floor((i + overlap) * fs * t), j],
                                              fs)[1][start:end])
    ac_mean = all_sig_corr.mean(axis=0)
    ac_std = all_sig_corr.std(axis=0)
    return ac_mean, ac_std

def coherence(signal1, signal2, fs):
    """
        Calculate coherence between two signals up to a maximal lag.
    :param signal1: signal1: EMG signal of participant1
    :param signal2: EMG signal of participant2
    :param fs: sampling frequency of the measuring device
    :return: frequencies and coherence arrays for each frequency
    """
    f, Cxy = signal.coherence(signal1, signal2, fs=fs, nperseg=fs)
    return f, Cxy

def describe_std_mean_all_tasks_pearson(tasks_lst1, tasks_lst2):
    """
        calculates the STD and mean of Person r of the signals of
        two participants during each task and between tasks
        :param tasks_lst1: a list of csv paths that contain the signal of participant1
        :param tasks_lst2: a list of csv paths that contain the signal of participant2
        """
    fs = 4000
    mean_std_per_task = np.zeros((len(tasks_lst1), 17, 2))
    # print("Pearson statistics")
    for i in range(len(tasks_lst1)):
        signal1, signal2 = breath_preproccess_signals(tasks_lst1[i], tasks_lst2[i], 'norm')
        ac_mean, ac_std = get_mean_std_of_pearson(signal1, signal2, fs)
        # print(f"{tasks_lst1[i]} - mean = {ac_mean}, std = {ac_std}")
        for j in range(np.shape(signal1)[1]):
            mean_std_per_task[i, j, 0] = ac_mean[j]
            mean_std_per_task[i, j, 1] = ac_std[j]
    for i in range(17):
        fig, ax = plt.subplots(figsize=(16,8))
        ax.bar(range(12), mean_std_per_task[:, i, 0], yerr=mean_std_per_task[:, i, 1], align='center', alpha=0.5,
               ecolor='black', capsize=10)
        ax.set_ylabel('Pearson r', fontsize=20)
        ax.set_xlabel('tasks', fontsize=20)
        plt.rcParams['font.size'] = '22'
        ax.set_xticks(range(12))
        ax.set_xticklabels(['trig', 'audio1', 'between', 'blink', 'between', 'button',
                            'between', 'mirror', 'between', 'smile', 'between', 'audio2'], fontsize=16)
        fig.show()
        fig.savefig(fr'Pearson r mean and std of each task in channel {i}')
    plt.close()

def breath_describe_std_mean_all_tasks_pearson(tasks_lst1, tasks_lst2):
    """
        calculates the STD and mean of Person r of the signals of
        two participants during each task and between tasks
        :param tasks_lst1: a list of csv paths that contain the signal of participant1
        :param tasks_lst2: a list of csv paths that contain the signal of participant2
        """
    fs = 20
    mean_std_per_task = np.zeros((len(tasks_lst1), 3, 2))
    # print("Pearson statistics")
    for i in range(len(tasks_lst1)):
        signal1, signal2 = breath_preproccess_signals(tasks_lst1[i], tasks_lst2[i], 'norm')
        ac_mean, ac_std = get_mean_std_of_pearson(signal1, signal2, fs)
        # print(f"{tasks_lst1[i]} - mean = {ac_mean}, std = {ac_std}")
        for j in range(np.shape(signal1)[1]):
            mean_std_per_task[i, j, 0] = ac_mean[j]
            mean_std_per_task[i, j, 1] = ac_std[j]
    for i in range(3):
        fig, ax = plt.subplots(figsize=(16,8))
        ax.bar(range(12), mean_std_per_task[:, i, 0], yerr=mean_std_per_task[:, i, 1], align='center', alpha=0.5,
               ecolor='black', capsize=10)
        ax.set_ylabel('Pearson r', fontsize=20)
        ax.set_xlabel('tasks', fontsize=20)
        plt.rcParams['font.size'] = '22'
        ax.set_xticks(range(12))
        ax.set_xticklabels(['trig', 'audio1', 'between', 'blink', 'between', 'button',
                            'between', 'mirror', 'between', 'smile', 'between', 'audio2'], fontsize=16)
        fig.show()
        fig.savefig(fr'breath Pearson r mean and std of each task in channel {i}')
    plt.close()

def get_mean_std_of_pearson(signal1, signal2, fs, blink_trig_time=7, overlap=4):
    """
        calculates Pearson r between two signals in each window and returns the mean and STD of all the windows
        :param signal1: EMG signal of participant1
        :param signal2: EMG signal of participant2
        :param fs: sampling frequency of the measuring device
        :param blink_trig_time: length of the blinking trigger and window to calculate
        :param overlap: overlap between each two windows
        :return: mean and STD of Pearson r in all the windows
        """
    t = blink_trig_time / overlap
    all_sig_corr = np.zeros((math.floor(len(signal1) / (fs * t)) - overlap + 1, np.shape(signal1)[1]))
    for i in range(math.floor(len(signal1) / (fs * t))-overlap+1):
        for j in range(np.shape(signal1)[1]):
            all_sig_corr[i, j] = pearson(signal1[math.floor(i * fs * t):math.floor((i + overlap) * fs * t), j],
                                         signal2[math.floor(i * fs * t):math.floor((i + overlap) * fs * t), j],)[1]
    ac_mean = all_sig_corr.mean(axis=0)
    ac_std = all_sig_corr.std(axis=0)
    return ac_mean, ac_std

def pearson(signal1, signal2):
    """
        calculates Pearson r of two signals in a given window
    :param signal1: EMG signal of participant1
    :param signal2: EMG signal of participant2
    :return: Pearson r of two signals in a given window
    """
    return stats.pearsonr(signal1, signal2)

def blink_analysis(tasks_lst1, tasks_lst2):
    """
        calculates the blinking synchronization of the signals of
        two participants during each task and between tasks
    :param tasks_lst1: a list of csv paths that contain the signal of participant1
    :param tasks_lst2: a list of csv paths that contain the signal of participant2
    """
    fs = 4000
    labels = ['trig', 'audio1', 'between', 'blink', 'between', 'button',
                            'between', 'mirror', 'between', 'smile', 'between', 'audio2']
    rms_lst = []
    signal1, signal2 = breath_preproccess_signals(tasks_lst1[0], tasks_lst2[0], 'norm_stand')
    EEG1 = signal1[:,15]
    EEG2 = signal2[:,15]
    blinking_pattern1 = create_blinking_filter(EEG1[14000:15500], 3)
    blinking_pattern2 = create_blinking_filter(EEG2[7800:10200], 3)
    for i in range(len(tasks_lst1)):
        EEG1, EEG2 = breath_preproccess_signals(tasks_lst1[i], tasks_lst2[i], 'norm_stand')
        peaks_height1 = 0.05
        peaks_height2 = 0.02
        spike_train1 = convert_patten_to_spike_train(EEG1[:,15], blinking_pattern1, fs, peaks_height1)
        spike_train2 = convert_patten_to_spike_train(EEG2[:,15], blinking_pattern2, fs, peaks_height2)
        rms = preprocess_func.calc_RMS(spike_train1, spike_train2, fs)
        rms_lst.append(rms)
    arr = np.array(rms_lst)
    plt.figure(figsize=(10,6))
    plt.bar(range(12), arr)
    plt.ylabel('RMS')
    plt.xticks(range(12), labels)
    plt.title('RMS in each trial')
    plt.savefig('RMS in each trial channel 15')
    plt.show()

def create_blinking_filter(sig, fft_order):
    """
        creates a filter of a pattern of a participant using FFT in order to recognize it in a signal
    :param sig: EMG signal of a participant
    :param fft_order: the requested number of Fourier coefficients to calculate for the filter
    :return: a filter of a pattern
    """
    n = len(sig)
    fftsig = np.fft.fft(sig)
    np.put(fftsig, range(fft_order + 1, n), 0.0)
    win = np.fft.ifft(fftsig)
    win1 = np.real(np.fft.ifft(fftsig)) / np.sqrt(np.sum(np.square(win)))
    # plt.figure()
    # plt.plot(win1)
    # plt.title(fr"blinking pattern filter")
    # plt.show()
    return win1

def convert_patten_to_spike_train(sig, pattern, fs, peaks_height):
    """
        gets an EMG signal and a blinking pattern filter and converts the signal to a spike train of blinks
    :param sig: EMG signal of a participant
    :param filter: a blinking pattern filter
    :param fs: sampling frequency of the measuring device
    :param peaks_height: peaks height to recognize after correlation
    :return: a spike train of blinks
    """
    N = len(sig)
    t = np.linspace(1 / fs, N, N)
    corr = signal.correlate(sig, pattern, mode='same') / len(pattern)
    peaks = signal.find_peaks(corr, height=peaks_height)
    spike_train = np.zeros_like(t)
    spike_train[peaks[0]] = 1
    # plt.figure()
    # plt.title("recognize blinks")
    # plt.plot(t, np.real(corr))
    # plt.plot(peaks[0], peaks[1]['peak_heights'], '*')
    # plt.show()
    # plt.close()
    # plt.figure()
    # plt.title("The created spike train")
    # plt.stem(t, spike_train)
    # plt.show()
    # plt.close()
    return spike_train