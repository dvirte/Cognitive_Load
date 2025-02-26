import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import title


def create_heatmap_top_features(
        subject_ids,
        data_dir,
        label_col="Difficulty Rating",
        top_k=2,
        remove_high_corr=True,
        corr_threshold=0.85,
        output_file=None
):
    """
    Creates a heatmap of feature-vs-subject correlation with NASA-TLX.

    Parameters
    ----------
    subject_ids : list of str
        IDs for each participant (e.g. ['01', '02', '03', ...]).
    data_dir : str
        Path to the folder containing subfolders like 'participant_{id}/S01/feature_table.csv'.
    label_col : str
        Name of the NASA-TLX rating column in the feature tables.
    top_k : int
        Number of top-correlated features to select for each subject.
    remove_high_corr : bool
        If True, filters out features that are highly correlated with each other (>= corr_threshold).
    corr_threshold : float
        Threshold for removing features that are highly correlated with each other.
    output_file : str or None
        If not None, the function will save the heatmap to this filename.

    Returns
    -------
    None. (Displays or saves the heatmap.)
    """
    # ---------------------------------------------------------
    # 1) Collect each subject's feature table
    #    (We assume each subject folder has 'S01/feature_table.csv'.)
    # ---------------------------------------------------------
    subject_dfs_raw = {}
    for sid in subject_ids:
        csv_path = os.path.join(data_dir, f"participant_{sid}", "S01", "feature_table.csv")
        if not os.path.exists(csv_path):
            print(f"[WARNING] Missing feature table for subject {sid}: {csv_path} not found.")
            continue
        df = pd.read_csv(csv_path)

        # Keep only numeric columns + the label
        # (Ignore 'Period' or others if you wish.)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if label_col in numeric_cols:
            # We'll keep it in for correlation computations
            pass
        else:
            raise ValueError(f"Label column '{label_col}' not in numeric columns for subject {sid}.")

        subject_dfs_raw[sid] = df

    # ---------------------------------------------------------
    # 2) (Optional) For each subject, remove highly correlated features
    # ---------------------------------------------------------
    # We'll define a small helper that removes correlated features,
    # keeping the one with higher correlation to the label.
    def filter_correlated_features(df_sub, label_col, corr_thr):
        # We'll isolate numeric columns (aside from label).
        all_numeric = df_sub.select_dtypes(include=[np.number]).columns
        remove_f=[label_col, 'Period']
        if label_col not in all_numeric:
            return df_sub  # or raise error
        features_only = [c for c in all_numeric if c not in remove_f]

        corr_matrix = df_sub[features_only].corr().abs()
        label_corr = df_sub[features_only].corrwith(df_sub[label_col]).abs().sort_values(ascending=False)

        to_remove = set()
        for i in range(len(label_corr)):
            f_i = label_corr.index[i]
            if f_i in to_remove:
                continue
            for j in range(i + 1, len(label_corr)):
                f_j = label_corr.index[j]
                if f_j in to_remove:
                    continue
                if corr_matrix.loc[f_i, f_j] >= corr_thr:
                    # Remove the one with lower correlation to label
                    if label_corr[f_i] >= label_corr[f_j]:
                        to_remove.add(f_j)
                    else:
                        to_remove.add(f_i)
                        break

        keep_features = [f for f in features_only if f not in to_remove]
        return df_sub[keep_features + [label_col]]

    subject_dfs = {}
    if remove_high_corr:
        for sid in subject_dfs_raw:
            df_original = subject_dfs_raw[sid]
            df_filtered = filter_correlated_features(df_original, label_col, corr_threshold)
            subject_dfs[sid] = df_filtered

    # ---------------------------------------------------------
    # 3) For each subject, find top_k features by |corr| w.r.t. label_col
    # ---------------------------------------------------------
    subject_top_features = {}
    for sid, df_sub in subject_dfs.items():
        # compute correlation of each numeric col with label
        numeric_cols = df_sub.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c != label_col]

        corrs = df_sub[feature_cols].corrwith(df_sub[label_col]).abs()
        # sort descending
        corrs_sorted = corrs.sort_values(ascending=False)

        # pick top_k
        top_features = corrs_sorted.index[:top_k].tolist()
        subject_top_features[sid] = top_features

    # Union of all top_k features across subjects
    all_top_features = set()
    for sid in subject_top_features:
        for feat in subject_top_features[sid]:
            all_top_features.add(feat)
    all_top_features = list(all_top_features)  # convert set->list

    # ---------------------------------------------------------
    # 4) Build a (Subjects x Features) matrix of correlation with NASA-TLX
    # ---------------------------------------------------------
    # We'll create a DataFrame with rows=subjects, cols=features.
    heatmap_df = pd.DataFrame(index=subject_ids, columns=all_top_features, dtype=float)

    for sid in subject_ids:
        if sid not in subject_dfs_raw:
            continue
        df_sub = subject_dfs_raw[sid]

        # For each feature in all_top_features, compute correlation with label
        for feat in all_top_features:
            if feat not in df_sub.columns:
                heatmap_df.loc[sid, feat] = np.nan
            else:
                corr_val = df_sub[[feat, label_col]].corr().abs().iloc[0, 1]
                heatmap_df.loc[sid, feat] = corr_val

    # ---------------------------------------------------------
    # 5) Plot the heatmap
    # ---------------------------------------------------------
    # By default, correlation ranges from -1 to +1, so let's use a diverging colormap.
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_df,
        annot=True,  # show numeric correlation values in cells
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(f"Correlation (Feature vs. {label_col}) by Subject")
    plt.xlabel("Features")
    plt.ylabel("Subjects")
    plt.xticks(rotation=45, ha="right")  # rotate feature labels if they are long
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
    plt.show()


def create_heatmap_top_features_annotated(
        subject_ids,
        data_dir,
        label_col="Difficulty Rating",
        top_k=2,
        remove_high_corr=True,
        corr_threshold=0.85,
        output_file=None
):
    """
    Creates a heatmap of (Subject vs. Feature) correlation with NASA-TLX,
    and annotates any cell whose feature is highly correlated (>= corr_threshold)
    with another feature in the same subject's data.
    """
    # ---------------------------------------------------------
    # 1) Collect & optionally filter each subject's feature table
    # ---------------------------------------------------------
    subject_dfs_raw = {}
    for sid in subject_ids:
        csv_path = os.path.join(data_dir, f"participant_{sid}", "S01", "feature_table.csv")
        if not os.path.exists(csv_path):
            print(f"[WARNING] Missing feature table for subject {sid}: {csv_path} not found.")
            continue
        df = pd.read_csv(csv_path)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if label_col not in numeric_cols:
            raise ValueError(f"Label column '{label_col}' not in numeric columns for subject {sid}.")
        subject_dfs_raw[sid] = df

    def filter_correlated_features(df_sub, label_col, corr_thr):
        # Identify all numeric columns except 'Period' and the label
        remove_f = [label_col, 'Period']
        all_numeric = df_sub.select_dtypes(include=[np.number]).columns
        features_only = [c for c in all_numeric if c not in remove_f]

        # Pairwise feature-feature correlation
        corr_matrix = df_sub[features_only].corr().abs()
        # Feature-label correlation
        label_corr = df_sub[features_only].corrwith(df_sub[label_col]).abs().sort_values(ascending=False)

        to_remove = set()
        for i in range(len(label_corr)):
            f_i = label_corr.index[i]
            if f_i in to_remove:
                continue
            for j in range(i + 1, len(label_corr)):
                f_j = label_corr.index[j]
                if f_j in to_remove:
                    continue
                if corr_matrix.loc[f_i, f_j] >= corr_thr:
                    # Remove the one with the lower label correlation
                    if label_corr[f_i] >= label_corr[f_j]:
                        to_remove.add(f_j)
                    else:
                        to_remove.add(f_i)
                        break

        keep_features = [f for f in features_only if f not in to_remove]
        return df_sub[keep_features + [label_col]]

    subject_dfs = {}
    for sid, df_raw in subject_dfs_raw.items():
        df_filtered = filter_correlated_features(df_raw, label_col, corr_threshold)
        subject_dfs[sid] = df_filtered

    # ---------------------------------------------------------
    # 2) For each subject, find top_k features by |corr| w.r.t. label_col
    # ---------------------------------------------------------
    subject_top_features = {}
    for sid, df_sub in subject_dfs.items():
        numeric_cols = df_sub.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c != label_col]

        corrs = df_sub[feature_cols].corrwith(df_sub[label_col]).abs()
        corrs_sorted = corrs.sort_values(ascending=False)
        top_features = corrs_sorted.index[:top_k].tolist()
        subject_top_features[sid] = top_features

    # Create a union of all top_k features
    all_top_features = set()
    for sid in subject_top_features:
        for feat in subject_top_features[sid]:
            all_top_features.add(feat)
    all_top_features = list(all_top_features)

    # ---------------------------------------------------------
    # 3) Build a subject × feature matrix of correlation w/ NASA‐TLX
    #    ALSO build an annotation matrix that adds "*" if there's any
    #    same‐subject feature with correlation >= corr_threshold
    # ---------------------------------------------------------
    heatmap_df = pd.DataFrame(index=subject_ids, columns=all_top_features, dtype=float)
    annot_df = pd.DataFrame(index=subject_ids, columns=all_top_features, dtype=object)

    for sid in subject_ids:
        # We'll use the 'raw' DataFrame to measure correlation among all features:
        df_raw = subject_dfs_raw[sid]
        df_sub = df_raw

        # 3A. Build a pairwise correlation matrix among all features in df_sub
        numeric_cols_sub = df_sub.select_dtypes(include=[np.number]).columns
        features_sub_only = [c for c in numeric_cols_sub if c != label_col]
        pairwise_corr = df_sub[features_sub_only].corr().abs()

        # 3B. For each feature in all_top_features, compute correlation w/ NASA‐TLX
        for feat in all_top_features:
            corr_val = df_sub[[feat, label_col]].corr().iloc[0, 1]
            # We'll store the absolute correlation for the heatmap color
            corr_abs = abs(corr_val)
            heatmap_df.loc[sid, feat] = corr_abs

            # 3C. Check if this feature is correlated >= threshold with any other feature
            # within df_sub
            # If there's any other feature with correlation >= threshold
            # (besides itself) => add a star
            high_corr_mask = (pairwise_corr.loc[feat] >= corr_threshold) & (
                        pairwise_corr.loc[feat].index != feat)
            # We'll only consider the top features for annotation
            high_corr_mask = high_corr_mask[all_top_features]
            # Drop the current feature from the mask
            high_corr_mask = high_corr_mask.drop(feat, errors='ignore')

            if (high_corr_mask.any()
                    and feat not in subject_top_features[sid]
                    and corr_abs >= 0.75):
                # Annotate with star plus numeric correlation (rounded)
                annot_df.loc[sid, feat] = f"*{corr_abs:.2f}"
            else:
                annot_df.loc[sid, feat] = f"{corr_abs:.2f}"

    # ---------------------------------------------------------
    # 4) Rename columns (replacing underscores) & Plot
    # ---------------------------------------------------------
    # 4A) Rename columns by replacing underscores with spaces
    old_cols = heatmap_df.columns.tolist()
    new_cols = [c.replace('_', ' ') for c in old_cols]
    heatmap_df.columns = new_cols
    annot_df.columns = new_cols

    # Ensure index (feature names) and columns are correctly formatted as strings
    heatmap_df.index = heatmap_df.index.astype(str)
    heatmap_df.columns = heatmap_df.columns.astype(str)

    # **Sort features by their average correlation across all subjects**
    sorted_features = heatmap_df.mean(axis=0).sort_values(ascending=False).index

    # Reorder the heatmap and annotation dataframes based on sorted features
    heatmap_df_s = heatmap_df[sorted_features]
    annot_df_s = annot_df[sorted_features]

    # 1) Identify the top feature
    top_feature = sorted_features[0]

    # 2) Sort subjects (the DataFrame rows) by the value of that top feature, descending
    #    This will reorder the rows of heatmap_df_s so that the subject with the
    #    highest value in the top feature is first.
    heatmap_df_s = heatmap_df_s.sort_values(by=top_feature, ascending=False)

    # Keep annotation data aligned
    annot_df_s = annot_df_s.loc[heatmap_df_s.index]

    # 3) Now create the heatmap using the transposed data
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_df_s.T,
        annot=annot_df_s.T,
        fmt="",
        cmap="RdBu_r",
        center=0,
        cbar_kws={"shrink": 0.8, "label": "Correlation"}
    )

    plt.text(-1.5, -1, "Features", fontsize=12, ha="center", va="bottom")
    plt.xlabel("Subjects subjective report")
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
    plt.show()

    # ---------------------------------------------------------
    # 5) Create Box Plot of Correlation Distributions
    # ---------------------------------------------------------
    # A) Reshape the wide heatmap_df into a long format with columns ["Subject", "Feature", "Correlation"].
    #    Right now, heatmap_df has rows=Subjects, cols=Features
    #    So we'll do .reset_index() to add "Subject" as a column, then .melt().
    heatmap_df_box = heatmap_df.reset_index().rename(columns={"index": "Subject"})

    # Melt
    melted_df = heatmap_df_box.melt(
        id_vars="Subject",  # columns to keep as-is
        var_name="Feature",  # name for the new feature column
        value_name="Correlation"  # name for the new correlation value column
    )

    # Sort features by mean correlation
    feature_order = melted_df.groupby("Feature")["Correlation"].mean().sort_values(ascending=False).index

    # Plot sorted box plot with overlaid individual points
    plt.figure(figsize=(10, 12))
    ax = sns.boxplot(
        data=melted_df,
        y="Feature",
        x="Correlation",
        order=feature_order,  # Sorting applied here
        color="lightblue"
    )

    # Overlay individual points
    sns.stripplot(
        data=melted_df,
        y="Feature",
        x="Correlation",
        order=feature_order,  # Ensure sorting is consistent
        color="black",
        alpha=0.5,
        size=4  # Adjust point size if needed
    )

    # Set axis labels
    ax.set_xlabel("Absolute Correlation w/ NASA-TLX")
    ax.set_ylabel("Features", rotation=0, fontsize=12)
    ax.yaxis.set_label_coords(-0.15, 1.)

    plt.grid(axis='x', linestyle='--', alpha=0.7)  # Optional: add gridlines
    plt.tight_layout()
    plt.show()

    # plt.ylabel("Features")
    plt.ylabel("")

    # Calculate the y-position to place the label above the features
    y_pos = len(melted_df["Feature"].unique()) + 0.5  # Increase y_pos to push text above

    # Manually place "Features" label **above the feature names**
    plt.text(
        x=melted_df["Correlation"].min() - 0.05,  # Align with the leftmost x-value
        y=y_pos,
        s="Features",
        fontsize=14,
        fontweight="bold",
        ha="left",  # Align text to the left
        va="top"
    )

    plt.xlabel("Absolute Correlation w/ NASA-TLX")
    plt.grid(axis='x', linestyle='--', alpha=0.7)  # Optional: Add gridlines for readability
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 5) Create Plot of Correlation Distributions
    # ---------------------------------------------------------
    feature_name = subject_top_features['06']
    feature_name = feature_name+['Difficulty Rating']
    feature_value = subject_dfs_raw['06']
    feature_values = feature_value[feature_name]

    # Create range of number of rows
    rows = np.array(range(len(feature_values)))  # Update row indices

    #shape of the data
    print(feature_values.shape)
    # shape of the rows
    print(rows.shape)

    plt.figure(figsize=(10, 6))
    for feature in feature_name:
        val = np.array(feature_values[feature])
        # normalize the values by standard deviation and mean
        val = (val - np.mean(val)) / np.std(val)

        # Format feature name for legend
        formatted_feature = feature.replace('_', ' ')

        if feature == 'Difficulty Rating':
            # plot scatter plot using 'x'
            plt.scatter(rows, val, label=feature, marker='x')
        else:
            plt.scatter(rows, val, label=formatted_feature)
            plt.plot(rows, val, linestyle='-')
    plt.xlabel("Maze Number")
    plt.ylabel("Normalized Values")
    plt.legend()
    plt.show()


    list_of_feat = ['beta_channel13_mean_band_power', 'beta_channel13_band_power_variance',
                    'beta_channel13_max_band_power', 'beta_channel13_min_band_power',
                    'beta_channel13_band_power_iqr']
    label_name = 'Difficulty Rating'
    for feature_beta in list_of_feat:
        plt.figure(figsize=(10, 6))
        for sid in subject_ids:
            df_sub = subject_dfs_raw[sid]
            val = np.array(df_sub[feature_beta])
            val = (val - np.mean(val)) / np.std(val)
            rows = np.array(range(len(val)))
            plt.scatter(rows, val, label=sid)
            plt.plot(rows, val, linestyle='-')
            # print correlation with label
            corr_val = df_sub[[feature_beta, label_name]].corr().iloc[0, 1]
            print(f"Feature {feature_beta} Correlation for {sid}: {corr_val:.2f}")
        plt.xlabel("Maze Number")
        plt.ylabel("Normalized Values")
        plt.title(f"Feature: {feature_beta}")
        plt.legend()
        plt.show()

