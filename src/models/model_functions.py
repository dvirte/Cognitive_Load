import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch,find_peaks
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.decomposition import PCA
from src.signal_processing.proccess_func import bandpass_filter, extract_rms_by_band
from src.visualization import vis_cog as vc

def extract_rms_features(rms_data, time_vec, features, band, channels, fs):
    for idx, ch in enumerate(channels):
        channel_data = rms_data[:, idx] # RMS data for the current channel
        t_total = time_vec[-1] - time_vec[0]    # Total duration in seconds

        # Statistical Features
        features[f"{band}_channel{ch + 1}_mean_rms"] = np.mean(channel_data)
        features[f"{band}_channel{ch + 1}_std_rms"] = np.std(channel_data)
        features[f"{band}_channel{ch + 1}_max_rms"] = np.max(channel_data)
        features[f"{band}_channel{ch + 1}_min_rms"] = np.min(channel_data)
        features[f"{band}_channel{ch + 1}_iqr_rms"] = np.percentile(channel_data, 75) - np.percentile(channel_data, 25)
        features[f"{band}_channel{ch + 1}_entropy_rms"] = -np.sum(channel_data * np.log2(channel_data + 1e-12))

        # Temporal Features
        rms_gradient = np.gradient(channel_data, time_vec)
        features[f"{band}_channel{ch + 1}_rms_slope"] = np.mean(rms_gradient)
        features[f"{band}_channel{ch + 1}_rms_gradient_variance"] = np.var(rms_gradient)
        features[f"{band}_channel{ch + 1}_max_gradient"] = np.max(rms_gradient)
        features[f"{band}_channel{ch + 1}_min_gradient"] = np.min(rms_gradient)
        features[f"{band}_channel{ch + 1}_gradient_range"] = np.max(rms_gradient) - np.min(rms_gradient)
        features[f"{band}_channel{ch + 1}_positive_gradient_count"] = np.sum(rms_gradient > 0)
        features[f"{band}_channel{ch + 1}_negative_gradient_count"] = np.sum(rms_gradient < 0)
        features[f"{band}_channel{ch + 1}_positive_gradient_ratio"] = np.sum(rms_gradient > 0) / len(rms_gradient)
        features[f"{band}_channel{ch + 1}_negative_gradient_ratio"] = np.sum(rms_gradient < 0) / len(rms_gradient)

        # Event-Based Features
        threshold = np.mean(channel_data) + 2 * np.std(channel_data)
        peaks = [r for r in channel_data if r > threshold]
        features[f"{band}_channel{ch + 1}_peaks_count_per_sec"] = len(peaks) / t_total
        features[f"{band}_channel{ch + 1}_mean_peak_height"] = np.mean(peaks) if peaks else 0

        # Dynamic Features
        high_rms_duration_sec  = np.sum(channel_data > np.mean(channel_data) + np.std(channel_data))/len(channel_data)
        features[f"{band}_channel{ch + 1}_high_rms_duration"] = high_rms_duration_sec
        features[f"{band}_channel{ch + 1}_activity_entropy"] = -np.sum(
            (channel_data / np.sum(channel_data)) * np.log2((channel_data / np.sum(channel_data)) + 1e-12)
        )

    return features

def extract_frequency_features(raw_data, features, band, channels, fs, window_length=3, overlap=0.5):
    """
    Extract frequency-domain features from the signal using windowed PSD calculations.

    Parameters:
        raw_data (ndarray): Raw signal data, shape (n_samples, n_channels).
        features (dict): Dictionary to store the extracted features.
        band (str): Frequency band being processed (e.g., 'alpha', 'beta', 'emg').
        channels (iterable): Channels to process.
        fs (int): Sampling frequency of the signal.
        window_length (float): Window length in seconds for PSD.
        overlap (float): Overlap fraction for consecutive windows.

    Returns:
        features (dict): Updated dictionary with frequency-domain features.
    """
    # Band definitions for filtering
    band_ranges = {
        'alpha': (8, 13),
        'beta': (12, 30),
        'emg': (35, 124)
    }
    lowcut, highcut = band_ranges[band]

    # Calculate window size and step in samples
    window_size = min(int(window_length * fs), len(raw_data) - 1)
    window_step = int(window_size * (1 - overlap))

    for ch in channels:
        channel_data = raw_data[:, ch]

        # Split signal into overlapping windows
        n_windows = (len(channel_data) - window_size) // window_step + 1
        windowed_psd = []

        for i in range(n_windows):
            start = i * window_step
            end = start + window_size
            window = channel_data[start:end]

            # Calculate Welch's PSD for the current window
            freqs, psd = welch(window, fs=fs, nperseg=window_size)

            # Restrict PSD to the band of interest
            band_mask = (freqs >= lowcut) & (freqs <= highcut)
            band_psd = psd[band_mask]
            windowed_psd.append(np.sum(band_psd))  # Sum power in the band

        # Convert windowed PSD to numpy array for aggregation
        windowed_psd = np.array(windowed_psd)

        # Calculate frequency-domain features for the band
        features[f"{band}_channel{ch + 1}_mean_band_power"] = np.mean(windowed_psd)
        features[f"{band}_channel{ch + 1}_band_power_variance"] = np.var(windowed_psd)
        features[f"{band}_channel{ch + 1}_max_band_power"] = np.max(windowed_psd)
        features[f"{band}_channel{ch + 1}_min_band_power"] = np.min(windowed_psd)
        features[f"{band}_channel{ch + 1}_band_power_iqr"] = np.percentile(windowed_psd, 75) - np.percentile(
            windowed_psd, 25)

        # Temporal dynamics within the band
        features[f"{band}_channel{ch + 1}_early_band_power"] = np.mean(windowed_psd[:n_windows // 3])
        features[f"{band}_channel{ch + 1}_late_band_power"] = np.mean(windowed_psd[-n_windows // 3:])

        # Complexity metrics
        features[f"{band}_channel{ch + 1}_spectral_entropy"] = -np.sum(
            (windowed_psd / np.sum(windowed_psd + 1e-12)) * np.log2(windowed_psd / np.sum(windowed_psd + 1e-12) + 1e-12)
        )

    return features

def extract_time_features(raw_data, timestamps, features, band, channels, fs, threshold_factor=2,
    window_length=3.0, overlap=0.5):
    """
    Extract time-domain features in short windows, then aggregate them into single values.

    Parameters
    ----------
    raw_data : ndarray
        Signal data, shape (n_samples, n_channels). This can be raw or bandpass-filtered.
    timestamps : ndarray
        Timestamps corresponding to the signal samples, shape (n_samples,).
    features : dict
        Dictionary to store the aggregated features.
    band : str
        Frequency band name (e.g., 'alpha', 'beta', 'emg').
    channels : iterable
        Which channel indices to process.
    fs : int or float
        Sampling frequency in Hz.
    threshold_factor : float
        Multiplier for threshold-based peak detection within each window.
    window_length : float
        Length in seconds of each analysis window.
    overlap : float
        Overlap fraction (0.0 to 1.0) for consecutive windows.

    Returns
    -------
    features : dict
        Updated dictionary with aggregated time-domain features for each channel and band.
    """
    # Decide which channels to use based on band
    if band in ['alpha', 'beta']:
        # Channels 13-16 in 1-based => indices 12..15 in 0-based
        forehead_set = {12, 13, 14, 15}
        relevant_channels = sorted(forehead_set.intersection(channels))
    else:
        # e.g. 'emg' => use all channels (or whatever the user provided)
        relevant_channels = sorted(channels)

    # Basic parameters
    n_samples = raw_data.shape[0]
    total_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0

    # Calculate number of samples per window and step
    window_size = int(window_length * fs)
    step_size = int(window_size * (1 - overlap))
    if step_size < 1:
        step_size = 1  # Prevent zero or negative step in extreme overlap cases
    n_windows = max(0, (n_samples - window_size) // step_size + 1)

    # Define a helper function for easier aggregator naming
    def add_stat(name_prefix, array, ch):
        """
        For a given 'array' of window-level values, compute various statistics
        (mean, std, median, min, max, p10, p90) and add them to 'features' dict
        under a consistent naming scheme.

        name_prefix might be something like "absmean" or "peakcount".
        """
        if len(array) == 0:
            # If no windows, just store 0
            features[f"{band}_channel{ch}_wnd_{name_prefix}_mean"] = 0
            features[f"{band}_channel{ch}_wnd_{name_prefix}_std"] = 0
            features[f"{band}_channel{ch}_wnd_{name_prefix}_median"] = 0
            features[f"{band}_channel{ch}_wnd_{name_prefix}_min"] = 0
            features[f"{band}_channel{ch}_wnd_{name_prefix}_max"] = 0
            features[f"{band}_channel{ch}_wnd_{name_prefix}_p10"] = 0
            features[f"{band}_channel{ch}_wnd_{name_prefix}_p90"] = 0
            return

        features[f"{band}_channel{ch}_wnd_{name_prefix}_mean"] = np.mean(array)
        features[f"{band}_channel{ch}_wnd_{name_prefix}_std"] = np.std(array)
        features[f"{band}_channel{ch}_wnd_{name_prefix}_median"] = np.median(array)
        features[f"{band}_channel{ch}_wnd_{name_prefix}_min"] = np.min(array)
        features[f"{band}_channel{ch}_wnd_{name_prefix}_max"] = np.max(array)
        features[f"{band}_channel{ch}_wnd_{name_prefix}_p10"] = np.percentile(array, 10)
        features[f"{band}_channel{ch}_wnd_{name_prefix}_p90"] = np.percentile(array, 90)

    # Loop over channels
    for ch in relevant_channels:
        # Prepare lists to store the per-window calculations:
        window_absmeans = []
        window_variances = []
        window_skewnesses = []
        window_kurtoses = []
        window_amplranges = []
        window_peakcounts = []
        window_peakampls = []
        window_zcrs = []  # zero-crossing rates

        # Slide over the signal in windows
        channel_data = raw_data[:, ch]
        for w_idx in range(n_windows):
            start = w_idx * step_size
            end = start + window_size
            segment = channel_data[start:end]

            if len(segment) == 0:
                continue  # skip empty segments

            # 1. Basic amplitude-based stats
            absmean = np.mean(np.abs(segment))  # absolute mean
            var_ = np.var(segment)
            skew_ = skew(segment)
            kurt_ = kurtosis(segment)
            amp_range = np.max(segment) - np.min(segment)

            # 2. Peak detection in this window
            local_thresh = np.mean(segment) + threshold_factor * np.std(segment)
            peaks, _ = find_peaks(segment, height=local_thresh)
            peak_count = len(peaks)
            peak_ampl = np.mean(segment[peaks]) if peak_count > 0 else 0

            # 3. Zero-crossing rate in the window
            zc_rate = 0
            if len(segment) > 1:
                zero_crossings = np.sum(
                    np.diff(np.sign(segment - np.mean(segment))) != 0
                )
                zc_rate = zero_crossings / (len(segment) - 1)  # or len(segment)

            # Store the window-level results
            window_absmeans.append(absmean)
            window_variances.append(var_)
            window_skewnesses.append(skew_)
            window_kurtoses.append(kurt_)
            window_amplranges.append(amp_range)
            window_peakcounts.append(peak_count)
            window_peakampls.append(peak_ampl)
            window_zcrs.append(zc_rate)

        # ---------------------------------------------------------------------
        # Aggregate these window-level values into single (or multiple) scalars
        # ---------------------------------------------------------------------

        ch_name = ch + 1  # for 1-based labeling in feature names

        # Aggregator stats: mean, std, median, min, max, p10, p90
        add_stat("absmean", window_absmeans, ch_name)
        add_stat("variance", window_variances, ch_name)
        add_stat("skewness", window_skewnesses, ch_name)
        add_stat("kurtosis", window_kurtoses, ch_name)
        add_stat("amplrange", window_amplranges, ch_name)
        add_stat("peakcount", window_peakcounts, ch_name)
        add_stat("peakampl", window_peakampls, ch_name)
        add_stat("zcr", window_zcrs, ch_name)

    return features

def create_feature_table(processor, data, overlap=0.5):
    """
    Create a feature table with extracted features for each period.

    Parameters:
        processor: Object with `play_periods` and `extract_trials` methods.
        data: Object containing metadata (e.g., difficulty ratings in `trail_cog_nasa`).
        overlap (float): Overlap fraction for RMS computation.

    Returns:
        feature_table (pd.DataFrame): Feature table with features for each period.
    """
    # Define the frequency bands
    frequency_bands = {
        'alpha': (8, 13),
        'beta': (12, 30),
        'emg': (35, 124)
    }

    # Initialize feature table
    feature_rows = []

    # Compute RMS for each frequency band
    rms_results = extract_rms_by_band(processor, overlap=overlap)

    # Process each period
    for period_idx, period in enumerate(processor.play_periods):
        # Extract data for the current period
        trial_data = processor.extract_trials(status=1, period=period_idx)
        raw_data = trial_data[0]  # Signal data (n_samples x n_channels)
        timestamps = trial_data[1]  # Timestamps for the signal

        # Identify if this is a calibration period
        is_calibration = period_idx < 10
        period_type = "Calibration" if is_calibration else "Task"
        difficulty_rating = data.trail_cog_nasa[period_idx]  # Subject's difficulty rating

        # Initialize features dictionary for the period
        features = {
            "Period": period_idx + 1,
            "Type": period_type,
            "Difficulty Rating": difficulty_rating,
        }

        # Iterate through each frequency band
        for band, band_results in frequency_bands.items():
            # Select the appropriate channels for the band
            if band in ['alpha', 'beta']:
                channels = range(12, 16)  # EEG channels (13-16 in 1-based indexing)
            else:
                channels = range(raw_data.shape[1])  # All channels for EMG

            # Extract band-specific RMS data and timestamps
            rms_data = rms_results[band][period_idx]["rms"]
            time_vec = rms_results[band][period_idx]["time_vec"]

            # Call the feature extraction functions
            features = extract_rms_features(rms_data, time_vec, features, band, channels, processor.fs)
            features = extract_frequency_features(raw_data, features, band, channels, processor.fs)
            filtered_data = bandpass_filter(raw_data, band_results[0], band_results[1], processor.fs, order=4)
            features = extract_time_features(filtered_data, timestamps, features, band, channels, processor.fs)

        # Append the features for the current period
        feature_rows.append(features)

    # Convert to DataFrame
    feature_table = pd.DataFrame(feature_rows)

    return feature_table

def filter_highly_correlated_features(df, label_col="Difficulty Rating", corr_threshold=0.85):
    """
    Removes one feature from each pair of highly correlated features (>= corr_threshold),
    keeping the one that has higher correlation with the label.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame that includes the label_col and numeric feature columns.
    label_col : str
        The name of the label column to preserve in the data.
    corr_threshold : float
        Threshold above which two features are considered "highly correlated."

    Returns
    -------
    df_filtered : pd.DataFrame
        A DataFrame with the label_col plus the remaining (non-duplicate) features.
    """
    # 1. Separate out the label column (ensure it's in the DataFrame).
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

    # We'll keep the label aside, but preserve it in the final output.
    label_series = df[label_col]

    # 2. Drop non-numeric or irrelevant columns if needed, or just select numeric columns.
    #    Make sure to exclude the label from the correlation matrix below.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)  # don't want label in the correlation among features

    # Create a sub-DataFrame of only numeric feature columns
    df_features = df[numeric_cols]

    df_features = df_features.drop(columns="Period", errors="ignore")

    # 3. Compute the correlation matrix among the features
    corr_matrix = df_features.corr().abs()  # absolute correlation

    # 4. Sort features by their absolute correlation with the label (descending).
    #    We'll use this to decide which feature to keep if two features are correlated.
    label_corr = df_features.corrwith(label_series).abs().sort_values(ascending=False)

    # We'll track which features we *remove* in a set
    to_remove = set()

    # 5. For each pair above threshold, remove one feature
    #    (We can do an upper triangle scan to avoid double counting).
    #    We'll keep the feature that has the *higher* label correlation.
    for i in range(len(label_corr)):
        feature_i = label_corr.index[i]
        if feature_i in to_remove:
            # Already removed
            continue

        for j in range(i + 1, len(label_corr)):
            feature_j = label_corr.index[j]
            if feature_j in to_remove:
                continue

            # Check correlation between feature_i and feature_j
            if corr_matrix.loc[feature_i, feature_j] >= corr_threshold:
                # They are highly correlated.
                # Remove the one with the lower label correlation => keep the bigger label-corr.
                if label_corr[feature_i] >= label_corr[feature_j]:
                    # remove j
                    to_remove.add(feature_j)
                else:
                    # remove i
                    to_remove.add(feature_i)
                    break  # feature_i is removed, no need to compare it further

    # 6. Construct the final filtered DataFrame
    remaining_features = [f for f in numeric_cols if f not in to_remove]

    # Re-create a DataFrame with the remaining features plus the label
    df_filtered = pd.concat([df[remaining_features], label_series], axis=1)

    return df_filtered

def get_top_correlated_features(df, label_col="Difficulty Rating", top_n=20, plot_corr=False, save_path=None):
    """
    Return the top_n features most correlated with the label_col in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing all the features plus the label column.
    label_col : str, optional
        Name of the column that holds the target label/difficulty rating.
    top_n : int, optional
        How many of the most correlated features to return.
    plot_corr : bool, optional
        Whether to plot the correlation between the top features and the label.
    save_path : str, optional
        Path to save the plots if plot_corr is True.

    Returns
    -------
    top_feature_names : list of str
        Names of the features, sorted by descending correlation magnitude.
    top_correlations : ndarray
        The absolute correlation values corresponding to those features,
        in the same order.
    """
    # Filter highly correlated features
    df_filtered = filter_highly_correlated_features(df)

    # Drop non-numeric columns
    df_numeric = df_filtered.drop(columns=["Period"], errors="ignore")

    # 1. Compute the correlation matrix
    corr_matrix = df_numeric.corr()

    # 2. Get the series of correlations with the target label
    #    We take the absolute value so we can rank by magnitude (whether +/-).
    if label_col not in corr_matrix.columns:
        raise ValueError(f"Label column '{label_col}' not in DataFrame or not numeric.")

    label_corr = corr_matrix[label_col].abs()

    # 3. Remove the label_col itself from the series (so we don't get correlation of label w/ label)
    label_corr = label_corr.drop(label_col, errors='ignore')

    # 4. Sort by descending correlation magnitude
    label_corr_sorted = label_corr.sort_values(ascending=False)

    # 5. Take the top_n features
    top_features = label_corr_sorted.head(top_n)

    # 6. Extract the feature names and correlation values
    top_feature_names = top_features.index.tolist()
    top_correlations = top_features.values

    if plot_corr:
        # Plot the top_n features
        y1 = df[label_col]
        x = range(len(y1))
        for feature in top_feature_names:
            y2 = df[feature]
            # Normalize the feature values to the same scale as the label (min max scaling)
            y2 = (y2 - y2.min()) / (y2.max() - y2.min()) * (y1.max() - y1.min()) + y1.min()
            plt.figure(figsize=(8, 6))
            # Difficulties (y1)
            plt.scatter(x, y1, alpha=0.6, color='red', label=f"{label_col} (red)")
            plt.plot(x, y1, color='red', alpha=0.6, label='_nolegend_')

            # Feature (y2)
            plt.scatter(x, y2, alpha=0.6, color='blue', label=feature)
            plt.plot(x, y2, color='blue', alpha=0.6, label='_nolegend_')

            # Title, labels
            plt.title(f"{feature} vs. {label_col} (correlation: {label_corr[feature]:.2f})")
            plt.xlabel("maze number")
            plt.ylabel("Value - Normalized")

            # Show legend for only the scatter points
            plt.legend(loc='upper right')

            plt.show(block=False)
            # Save the plot if save_path is provided
            if save_path is not None:
                plt.savefig(save_path[:-3] + f"{feature}_vs_{label_col}.png")
                # Close the plot
                plt.close()

    return top_feature_names, top_correlations

def lasso_feature_selection(df, candidate_features, label_col="Difficulty Rating", n_splits=5,
    random_state=42, alphas=None, use_loo=False):
    """
    Selects features using Lasso with cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing both the candidate features and the label_col.
    candidate_features : list of str
        List of feature column names (e.g., your top-20 correlated features).
    label_col : str
        The column name of the target/label (numerical) in `df`.
    n_splits : int, optional
        Number of folds in KFold cross-validation (ignored if use_loo=True).
    random_state : int, optional
        Random seed for reproducibility in KFold.
    alphas : array-like, optional
        List of alpha values for Lasso to try.
        If None, a default log-spaced range is used.
    use_loo : bool, optional
        If True, use Leave-One-Out CV instead of KFold.

    Returns
    -------
    selected_features : list of str
        Names of the features that have non-zero coefficients in the final Lasso model.
    best_alpha : float
        The alpha chosen by LassoCV.
    lasso_model : LassoCV
        The fitted LassoCV model (you can inspect coefficients, etc.).
    """

    # 1. Subset DataFrame to candidate features + label
    #    We'll drop rows with NaN if necessary, or you can handle them differently.
    df_sub = df[candidate_features + [label_col]].dropna()

    # Separate X and y
    X = df_sub[candidate_features].values
    y = df_sub[label_col].values

    # normalize X to have zero mean and unit variance (important for Lasso)
    X = StandardScaler().fit_transform(X)

    # 2. Define cross-validation approach
    if use_loo:
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # 3. Define alpha grid if not provided
    if alphas is None:
        # For small data, you can try a modest range.
        # Or you can broaden it if you want more exhaustive search.
        alphas = np.logspace(-3, 1, 50)  # from 0.001 to 10, in log scale

    # 4. Fit LassoCV
    lasso_cv = LassoCV(
        alphas=alphas,
        cv=cv,
        max_iter=10000,
        random_state=random_state
    )
    lasso_cv.fit(X, y)

    # 5. Retrieve the best alpha
    best_alpha = lasso_cv.alpha_

    # 6. Identify which features are non-zero
    coefs = lasso_cv.coef_
    selected_features = [
        feat for feat, coef in zip(candidate_features, coefs)
        if abs(coef) > 1e-10
    ]

    return selected_features, best_alpha, lasso_cv

def build_and_evaluate_model(
    df,
    candidate_features,
    label_col="Difficulty Rating",
    use_loo=False,
    n_splits=5,
    random_state=42,
    alphas=None,
    retrain_lasso=True
):
    """
    1) Select features via Lasso-based feature selection.
    2) Build a model using only those selected features.
    3) Evaluate the model with cross-validation and return some performance metrics.

    Parameters
    ----------
    df : pd.DataFrame
        The full DataFrame containing all potential features + the label.
    candidate_features : list of str
        List of columns to consider in Lasso feature selection (e.g., your top 20).
    label_col : str, default="Difficulty Rating"
        The name of the target column.
    use_loo : bool, default=False
        If True, use Leave-One-Out CV inside Lasso. Otherwise uses KFold(n_splits).
    n_splits : int, default=5
        Number of splits for KFold (ignored if use_loo=True).
    random_state : int, default=42
        For reproducibility in KFold shuffling.
    alphas : array-like, optional
        List of alpha values for Lasso to try. If None, default logspace is used.
    retrain_lasso : bool, default=False
        If True, retrain a Lasso model with the best alpha.
        If False, we use a simple LinearRegression on the selected features.

    Returns
    -------
    model : estimator
        The final trained model (either Lasso or LinearRegression).
    selected_features : list of str
        The subset of features that were selected by Lasso.
    performance_dict : dict
        A dictionary containing mean/stdev of R^2 and MSE across CV.
    """

    # 1) Call your Lasso-based selection function
    selected_features, best_alpha, lasso_cv_model = lasso_feature_selection(
        df=df,
        candidate_features=candidate_features,
        label_col=label_col,
        n_splits=n_splits,
        random_state=random_state,
        alphas=alphas,
        use_loo=use_loo
    )

    # 2) Subset the DataFrame to the selected features + label
    df_sub = df[selected_features + [label_col]].dropna()
    X_sub = df_sub[selected_features].values
    # normalize X to have zero mean and unit variance (important for Lasso)
    X_sub = StandardScaler().fit_transform(X_sub)
    y_sub = df_sub[label_col].values

    # take the first 4 components
    pca = PCA(n_components=4)
    X_sub = pca.fit_transform(X_sub)


    # 3) Build final model
    if retrain_lasso:
        # We'll retrain Lasso using the best alpha found
        final_model = Lasso(alpha=best_alpha, max_iter=10000, random_state=random_state)
    else:
        # We'll build a simple linear regression model
        final_model = LinearRegression()

    # 4) Evaluate with cross-validation on these selected features
    #    We'll compute R^2 and MSE for demonstration
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # R^2 scoring
    r2_scores = cross_val_score(final_model, X_sub, y_sub, cv=cv, scoring='r2')
    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)

    # MSE scoring (neg_mean_squared_error returns negative MSE => we multiply by -1)
    neg_mse_scores = cross_val_score(final_model, X_sub, y_sub, cv=cv, scoring='neg_mean_squared_error')
    mse_scores = -neg_mse_scores
    mse_mean = np.mean(mse_scores)
    mse_std = np.std(mse_scores)

    # 5) Optionally, fit the final model on the entire subset once
    final_model.fit(X_sub, y_sub)

    # 6) Create a dict of performance metrics
    performance_dict = {
        "r2_mean": r2_mean,
        "r2_std": r2_std,
        "mse_mean": mse_mean,
        "mse_std": mse_std
    }

    return final_model, selected_features, performance_dict


def run_pipeline(processor, data, session_folder):
    if 'feature_table.csv' not in os.listdir(session_folder):
        feature = create_feature_table(processor, data)
        # Save the feature table
        feature.to_csv(os.path.join(session_folder, "feature_table.csv"), index=False)
    else:
        # Load the feature table from csv
        feature_table_path = os.path.join(session_folder, "feature_table.csv")
        feature = pd.read_csv(feature_table_path)

    top_feature_names, top_correlations = get_top_correlated_features(feature, plot_corr=False,
                                                                         save_path=session_folder)

    final_model, selected_features, performance_dict = build_and_evaluate_model(feature, top_feature_names)

    return final_model, selected_features, performance_dict, top_feature_names, top_correlations