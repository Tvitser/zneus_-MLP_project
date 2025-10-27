import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import traceback

warnings.filterwarnings('ignore')


class SpeedDatingEDA:
    def __init__(self, data, auto_clean=True, drop_threshold=0.5, outlier_z=4.0, verbose=False):
        """
        Initialize the EDA class with the speed dating dataset

        Parameters:
        data (pd.DataFrame): The speed dating dataset
        auto_clean (bool): Whether to run basic cleaning automatically on init
        drop_threshold (float): Fraction of missing values above which a column is dropped
        outlier_z (float): z-score threshold used to optionally remove extreme outliers
        verbose (bool): If True, print extra debug information during cleaning
        """
        self.raw = data.copy()
        self.data = data.copy()
        self.numeric_columns = None
        self.categorical_columns = None

        # Cleaning parameters
        self.drop_threshold = drop_threshold
        self.outlier_z = outlier_z
        self.verbose = verbose

        if auto_clean:
            try:
                self.clean_data()
            except Exception as e:
                # If cleaning fails, keep original data and surface a helpful message
                print("Error during automatic cleaning. Reverting to original data.")
                print("Cleaning exception:", e)
                traceback.print_exc()
                self.data = self.raw.copy()

    # ---------------------------
    # Cleaning / preprocessing
    # ---------------------------
    def clean_data(self):
        """Run data cleaning steps before analysis. Each step is applied safely so a single-column issue doesn't abort the whole pipeline."""
        # Wrap each major step so a failure in one doesn't stop others; log errors for debugging.
        steps = [
            ("strip_whitespace_and_quotes", self._strip_whitespace_and_quotes),
            ("replace_placeholders_with_nan", self._replace_placeholders_with_nan),
            ("convert_range_strings_to_midpoint", self._convert_range_strings_to_midpoint),
            ("coerce_numeric_like_columns", self._coerce_numeric_like_columns),
            ("standardize_categoricals", self._standardize_categoricals),
            ("drop_high_missing_cols", lambda: self._drop_high_missing_cols(self.drop_threshold)),
            ("remove_duplicates", lambda: self.data.drop_duplicates(inplace=True)),
            ("coerce_binary_columns", lambda: self._coerce_binary_columns(['match', 'decision', 'decision_o'])),
            ("simple_impute", self._simple_impute),
            ("remove_outliers_zscore", lambda: self._remove_outliers_zscore(self.outlier_z)),
        ]

        for name, func in steps:
            try:
                if self.verbose:
                    print(f"Running cleaning step: {name}")
                func()
            except Exception as e:
                print(f"Warning: cleaning step '{name}' failed with error: {e!r}")
                if self.verbose:
                    traceback.print_exc()
                # continue to next step

        # Re-identify column types after cleaning
        self.identify_column_types()

    def _strip_whitespace_and_quotes(self):
        """Trim strings and remove surrounding quotes from object columns."""
        obj_cols = self.data.select_dtypes(include=['object']).columns
        for col in obj_cols:
            try:
                # operate only on non-null entries
                ser = self.data[col]
                # Convert to string only for entries that are not already numeric and are not null
                # Using vectorized operations for speed and safety
                mask_notnull = ser.notna()

                # Convert to str for string ops, but keep original for nulls
                ser_str = ser.astype(str)
                ser_str = ser_str.str.strip()
                ser_str = ser_str.str.strip('\'"')

                # Make empty strings NaN
                ser_str = ser_str.replace('', np.nan)

                # Assign back only where we operated (preserving potential numeric types elsewhere)
                self.data.loc[mask_notnull, col] = ser_str[mask_notnull].values

            except Exception:
                if self.verbose:
                    print(f"Failed to strip whitespace/quotes for column: {col}")
                    traceback.print_exc()
                # leave the column as-is

    def _replace_placeholders_with_nan(self):
        """Replace common placeholder tokens with NaN."""
        placeholders = {'?': np.nan, '??': np.nan, 'n/a': np.nan, 'na': np.nan, 'None': np.nan, 'none': np.nan,
                        'NULL': np.nan, 'null': np.nan, '...': np.nan, '[...]': np.nan}
        obj_cols = self.data.select_dtypes(include=['object']).columns
        for col in obj_cols:
            try:
                self.data[col] = self.data[col].replace(placeholders)
            except Exception:
                if self.verbose:
                    print(f"Failed to replace placeholders for column: {col}")
                    traceback.print_exc()

    def _convert_range_strings_to_midpoint(self):
        """
        Convert strings like '[4-6]' or '4-6' to their numeric midpoint (5.0).
        Also handles '[21-100]' etc.
        """
        range_re = re.compile(r'^\[?\s*([+-]?\d+\.?\d*)\s*[-–]\s*([+-]?\d+\.?\d*)\s*\]?$')
        obj_cols = self.data.select_dtypes(include=['object']).columns
        for col in obj_cols:
            try:
                ser = self.data[col]

                def _parse_range(x):
                    if pd.isna(x):
                        return x
                    s = str(x).strip()
                    m = range_re.match(s)
                    if m:
                        a = float(m.group(1))
                        b = float(m.group(2))
                        return (a + b) / 2.0
                    return x

                self.data[col] = ser.map(_parse_range)
            except Exception:
                if self.verbose:
                    print(f"Failed to parse ranges in column: {col}")
                    traceback.print_exc()

    def _coerce_numeric_like_columns(self):
        """
        Attempt to convert object columns that are numeric-like into numeric dtype.
        Rules:
          - remove commas
          - remove trailing/leading % and convert to proportion if the majority of values parse
          - try pd.to_numeric(errors='coerce') and convert column if a reasonable fraction becomes numeric
        This function is defensive: it skips columns where conversion would make things worse.
        """
        obj_cols = self.data.select_dtypes(include=['object']).columns
        for col in obj_cols:
            try:
                ser_orig = self.data[col]
                # operate on non-null entries
                mask_notnull = ser_orig.notna()
                if mask_notnull.sum() == 0:
                    continue

                ser = ser_orig.astype(str).str.replace(',', '', regex=False).str.strip()

                # handle percentages
                is_pct = ser.str.contains('%', na=False)
                if is_pct.any():
                    ser_pct = ser.str.replace('%', '', regex=False)
                    coerced_pct = pd.to_numeric(ser_pct, errors='coerce')
                    success_rate = coerced_pct.notna().sum() / len(coerced_pct)
                    if success_rate > 0.5:
                        # convert and normalize percent to [0,1]
                        self.data[col] = coerced_pct / 100.0
                        if self.verbose:
                            print(f"Column '{col}' converted from percent-like to numeric with success_rate={success_rate:.2f}")
                        continue

                coerced = pd.to_numeric(ser, errors='coerce')
                success_rate = coerced.notna().sum() / len(coerced)
                if success_rate > 0.6:
                    # good candidate: convert entire column
                    self.data[col] = coerced
                    if self.verbose:
                        print(f"Column '{col}' coerced to numeric (success_rate={success_rate:.2f})")
                else:
                    # revert obvious 'nan' strings to proper NaN
                    self.data.loc[self.data[col].astype(str).isin(['nan', 'NaN', 'None', 'none']), col] = np.nan

            except Exception:
                if self.verbose:
                    print(f"Failed numeric coercion for column: {col}")
                    traceback.print_exc()

    def _standardize_categoricals(self):
        """Apply basic categorical normalization for common columns."""
        # Gender normalization (defensive)
        if 'gender' in self.data.columns:
            try:
                ser = self.data['gender'].astype(str).str.lower().str.strip()
                ser = ser.replace({'femal': 'female', 'f': 'female', 'malee': 'male', 'm': 'male'})
                # remove non-letter characters
                ser = ser.str.replace(r'[^a-z]', '', regex=True)
                # unknowns -> NaN
                ser = ser.where(ser.isin(['male', 'female']), np.nan)
                self.data['gender'] = ser
            except Exception:
                if self.verbose:
                    print("Failed to standardize 'gender' column")
                    traceback.print_exc()

        # Lowercase common text columns to reduce cardinality noise
        for col in ['field', 'race', 'race_o']:
            if col in self.data.columns and self.data[col].dtype == 'object':
                try:
                    s = self.data[col].astype(str).str.lower().str.strip()
                    s = s.replace({'nan': np.nan})
                    self.data[col] = s
                except Exception:
                    if self.verbose:
                        print(f"Failed to standardize categorical column: {col}")
                        traceback.print_exc()

    def _drop_high_missing_cols(self, threshold=0.5):
        """Drop columns with fraction of missing values above threshold."""
        missing_frac = self.data.isnull().mean()
        to_drop = missing_frac[missing_frac > threshold].index.tolist()
        if to_drop:
            print(f"Dropping columns with >{threshold*100:.0f}% missing values: {to_drop}")
            self.data.drop(columns=to_drop, inplace=True)

    def _coerce_binary_columns(self, column_candidates):
        """Try to coerce some known binary columns to 0/1 integers."""
        for col in column_candidates:
            if col in self.data.columns:
                try:
                    ser = self.data[col]
                    # If already numeric, try safe rounding
                    if pd.api.types.is_numeric_dtype(ser):
                        coerced = pd.to_numeric(ser, errors='coerce').round().astype('Int64')
                        self.data[col] = coerced
                        continue

                    lower = ser.astype(str).str.lower().str.strip()
                    mapping = {'yes': 1, 'no': 0, 'y': 1, 'n': 0, 'true': 1, 'false': 0, '1': 1, '0': 0}
                    mapped = lower.replace(mapping)
                    coerced = pd.to_numeric(mapped, errors='coerce')
                    success_rate = coerced.notna().sum() / len(coerced)
                    if success_rate > 0.4:
                        self.data[col] = coerced.astype('Int64')
                        if self.verbose:
                            print(f"Column '{col}' coerced to binary-like (success_rate={success_rate:.2f})")
                except Exception:
                    if self.verbose:
                        print(f"Failed binary coercion for column: {col}")
                        traceback.print_exc()

    def _simple_impute(self):
        """
        Simple imputation:
          - numeric columns -> median
          - categorical/object columns -> mode
        This is intentionally conservative (keeps dtype safety).
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        obj_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in numeric_cols:
            try:
                if self.data[col].isnull().any():
                    median = self.data[col].median()
                    self.data[col] = self.data[col].fillna(median)
            except Exception:
                if self.verbose:
                    print(f"Failed numeric imputation for column: {col}")
                    traceback.print_exc()

        for col in obj_cols:
            try:
                if self.data[col].isnull().any():
                    mode = self.data[col].mode(dropna=True)
                    if not mode.empty:
                        self.data[col] = self.data[col].fillna(mode[0])
                    else:
                        self.data[col] = self.data[col].fillna('unknown')
            except Exception:
                if self.verbose:
                    print(f"Failed categorical imputation for column: {col}")
                    traceback.print_exc()

    def _remove_outliers_zscore(self, z_thresh=4.0):
        """
        Remove rows with extreme z-score for numeric columns.
        Default threshold is high (4.0) so we only remove very extreme cases.
        """
        if z_thresh is None or z_thresh <= 0:
            return

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            return

        # avoid constant columns
        valid_cols = [c for c in numeric_cols if self.data[c].std(ddof=0) > 0]
        if not valid_cols:
            return

        zscores = np.abs((self.data[valid_cols] - self.data[valid_cols].mean()) / self.data[valid_cols].std(ddof=0))
        extreme = (zscores > z_thresh).any(axis=1)
        n_extreme = int(extreme.sum())
        if n_extreme > 0:
            print(f"Removing {n_extreme} rows with |z| > {z_thresh} in numeric columns")
            self.data = self.data.loc[~extreme].reset_index(drop=True)

    # ---------------------------
    # Existing EDA methods (operate on cleaned self.data)
    # ---------------------------
    def basic_info(self):
        """Display basic information about the dataset"""
        print("=" * 50)
        print("BASIC DATASET INFORMATION")
        print("=" * 50)

        print(f"Dataset shape: {self.data.shape}")
        print(f"Number of rows: {self.data.shape[0]}")
        print(f"Number of columns: {self.data.shape[1]}")

        print("\nData types:")
        print(self.data.dtypes.value_counts())

        print("\nFirst 5 rows:")
        print(self.data.head())

        print("\nMissing values summary:")
        missing_info = pd.DataFrame({
            'Missing Count': self.data.isnull().sum(),
            'Missing Percentage': (self.data.isnull().sum() / len(self.data)) * 100
        })
        missing_data = missing_info[missing_info['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        if not missing_data.empty:
            print(missing_data)
        else:
            print("No missing values found!")

    def identify_column_types(self):
        """Identify numeric and categorical columns"""
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        if self.verbose:
            print("Numeric columns:", len(self.numeric_columns), self.numeric_columns)
            print("Categorical columns:", len(self.categorical_columns), self.categorical_columns)
        else:
            print("Numeric columns:", len(self.numeric_columns))
            print("Categorical columns:", len(self.categorical_columns))

        return self.numeric_columns, self.categorical_columns

    def target_variable_analysis(self, target_col='match'):
        """
        Analyze the target variable (match)

        Parameters:
        target_col (str): Name of the target column
        """
        if target_col not in self.data.columns:
            print(f"Target column '{target_col}' not found in dataset")
            return

        print("=" * 50)
        print("TARGET VARIABLE ANALYSIS")
        print("=" * 50)

        target_counts = self.data[target_col].value_counts(dropna=False)
        target_percentage = self.data[target_col].value_counts(normalize=True, dropna=True) * 100

        print("Target variable distribution:")
        # If there are NaNs, value_counts(dropna=False) will include them, but target_percentage will not show NaN row
        for val, count in target_counts.items():
            perc = target_percentage.get(val, np.nan)
            print(f"  {val}: {count} ({perc:.2f}%)" if not pd.isna(perc) else f"  {val}: {count} (NaN%)")

        # Plot target distribution
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        target_counts.plot(kind='bar', color=sns.color_palette('pastel', n_colors=len(target_counts)))
        plt.title('Target Variable Distribution')
        plt.xlabel('Match')
        plt.ylabel('Count')
        plt.xticks(rotation=0)

        plt.subplot(1, 2, 2)
        non_null_counts = target_counts.drop(index=[np.nan]) if np.nan in target_counts.index else target_counts
        if len(non_null_counts) > 0:
            plt.pie(non_null_counts.values, labels=non_null_counts.index, autopct='%1.1f%%',
                    colors=sns.color_palette('pastel', n_colors=len(non_null_counts)))
            plt.title('Match Percentage')
        else:
            plt.text(0.5, 0.5, 'No non-null values to plot', horizontalalignment='center')

        plt.tight_layout()
        plt.show()

    def demographic_analysis(self):
        """Analyze demographic variables"""
        print("=" * 50)
        print("DEMOGRAPHIC ANALYSIS")
        print("=" * 50)

        # Gender distribution
        if 'gender' in self.data.columns:
            print("\nGender Distribution:")
            gender_counts = self.data['gender'].value_counts(dropna=False)
            print(gender_counts)

            plt.figure(figsize=(15, 10))

            plt.subplot(2, 3, 1)
            gender_counts.plot(kind='bar', color=['lightpink', 'lightblue'])
            plt.title('Gender Distribution')
            plt.xticks(rotation=45)

        # Age analysis
        if 'age' in self.data.columns:
            # ensure numeric
            if not pd.api.types.is_numeric_dtype(self.data['age']):
                # attempt conversion; errors -> NaN
                self.data['age'] = pd.to_numeric(self.data['age'], errors='coerce')

            print(f"\nAge Statistics:")
            # Use safe formatting in case of NaN
            mean_age = self.data['age'].mean()
            median_age = self.data['age'].median()
            min_age = self.data['age'].min()
            max_age = self.data['age'].max()
            print(f"Mean age: {mean_age:.2f}" if not pd.isna(mean_age) else "Mean age: NaN")
            print(f"Median age: {median_age:.2f}" if not pd.isna(median_age) else "Median age: NaN")
            if not (pd.isna(min_age) or pd.isna(max_age)):
                print(f"Age range: {min_age} - {max_age}")
            else:
                print("Age range: NaN - NaN (no numeric age data)")

            plt.subplot(2, 3, 2)
            # avoid crash when all NaN
            if self.data['age'].dropna().empty:
                plt.text(0.5, 0.5, 'No numeric age data', horizontalalignment='center')
            else:
                self.data['age'].hist(bins=20, alpha=0.7, color='lightgreen')
            plt.title('Age Distribution')
            plt.xlabel('Age')
            plt.ylabel('Frequency')

        # Race analysis
        if 'race' in self.data.columns:
            print("\nRace Distribution:")
            race_counts = self.data['race'].value_counts()
            print(race_counts.head(10))  # Show top 10

            plt.subplot(2, 3, 3)
            race_counts.head(10).plot(kind='bar', color='lightcoral')
            plt.title('Top 10 Race Distribution')
            plt.xticks(rotation=45)

        # Field of study
        if 'field' in self.data.columns:
            print("\nTop 10 Fields of Study:")
            field_counts = self.data['field'].value_counts()
            print(field_counts.head(10))

            plt.subplot(2, 3, 4)
            field_counts.head(10).plot(kind='bar', color='lightyellow')
            plt.title('Top 10 Fields of Study')
            plt.xticks(rotation=45)

        # Same race preference
        if 'samerace' in self.data.columns:
            plt.subplot(2, 3, 5)
            self.data['samerace'].value_counts().plot(kind='bar', color=['lightgray', 'lightblue'])
            plt.title('Same Race Pairs')
            plt.xticks(rotation=0)

        # Age difference
        if 'd_age' in self.data.columns:
            plt.subplot(2, 3, 6)
            if not pd.api.types.is_numeric_dtype(self.data['d_age']):
                self.data['d_age'] = pd.to_numeric(self.data['d_age'], errors='coerce')
            if self.data['d_age'].dropna().empty:
                plt.text(0.5, 0.5, 'No numeric d_age data', horizontalalignment='center')
            else:
                self.data['d_age'].hist(bins=20, alpha=0.7, color='purple')
            plt.title('Age Difference Distribution')
            plt.xlabel('Age Difference')

        plt.tight_layout()
        plt.show()

    def preference_analysis(self):
        """Analyze preference-related variables"""
        print("=" * 50)
        print("PREFERENCE ANALYSIS")
        print("=" * 50)

        preference_cols = [col for col in self.data.columns if 'pref_' in col or 'importance_' in col]

        if not preference_cols:
            print("No preference columns found")
            return

        print("Preference-related columns found:")
        for col in preference_cols:
            print(f"  - {col}")

        # Select a subset of preference columns for visualization
        key_prefs = ['pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence',
                     'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests']

        available_prefs = [col for col in key_prefs if col in self.data.columns]

        if available_prefs:
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(available_prefs[:6], 1):
                plt.subplot(2, 3, i)
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    self.data[col].hist(bins=20, alpha=0.7)
                    plt.title(f'Distribution of {col}')
                    plt.xlabel('Preference Score')
                    plt.ylabel('Frequency')

            plt.tight_layout()
            plt.show()

            # Summary statistics
            print("\nPreference Statistics:")
            pref_stats = self.data[available_prefs].describe()
            print(pref_stats)

    def correlation_analysis(self, target_col='match'):
        """Analyze correlations between variables"""
        print("=" * 50)
        print("CORRELATION ANALYSIS")
        print("=" * 50)

        # Select only numeric columns
        numeric_data = self.data.select_dtypes(include=[np.number])

        if target_col in numeric_data.columns:
            # Correlation with target
            correlations = numeric_data.corr()[target_col].sort_values(ascending=False)

            print(f"Top 10 features most correlated with {target_col}:")
            print(correlations.head(10))

            print(f"\nBottom 10 features least correlated with {target_col}:")
            print(correlations.tail(10))

            # Correlation heatmap for top correlated features
            top_features = correlations.abs().sort_values(ascending=False).head(11).index
            corr_matrix = numeric_data[top_features].corr()

            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                        square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            plt.title('Correlation Heatmap of Top Features with Target')
            plt.tight_layout()
            plt.show()

        else:
            print(f"Target column '{target_col}' not found in numeric columns")

    def match_success_analysis(self, target_col='match'):
        """Analyze factors affecting match success"""
        if target_col not in self.data.columns:
            print(f"Target column '{target_col}' not found")
            return

        print("=" * 50)
        print("MATCH SUCCESS ANALYSIS")
        print("=" * 50)

        # Select key variables to analyze against match success
        key_vars = ['age', 'gender', 'samerace', 'importance_same_race',
                    'importance_same_religion', 'like', 'met']

        available_vars = [var for var in key_vars if var in self.data.columns]

        plt.figure(figsize=(15, 12))

        for i, var in enumerate(available_vars[:6], 1):
            plt.subplot(2, 3, i)

            if pd.api.types.is_numeric_dtype(self.data[var]):
                # For numeric variables, show distribution by match status
                try:
                    self.data.boxplot(column=var, by=target_col, ax=plt.gca())
                except Exception:
                    # fallback: simple grouped hist
                    grouped = self.data.dropna(subset=[var, target_col]).groupby(target_col)[var]
                    for label, g in grouped:
                        plt.hist(g, alpha=0.5, label=str(label))
                    plt.legend()
                plt.title(f'{var} by Match Status')
                plt.suptitle('')  # Remove automatic title
            else:
                # For categorical variables, show cross-tabulation
                try:
                    cross_tab = pd.crosstab(self.data[var], self.data[target_col], normalize='index')
                    cross_tab.plot(kind='bar', ax=plt.gca())
                except Exception:
                    # fallback: value_counts by group
                    vc = self.data.groupby(target_col)[var].value_counts(normalize=True).unstack(fill_value=0)
                    vc.plot(kind='bar', ax=plt.gca())
                plt.title(f'Match Rate by {var}')
                plt.xticks(rotation=45)

            plt.tight_layout()

        plt.show()

        # Statistical test for key variables
        print("\nStatistical significance of key variables with match success:")
        for var in available_vars[:5]:  # Test first 5 variables
            if pd.api.types.is_numeric_dtype(self.data[var]):
                # T-test for numeric variables
                match_1 = self.data[self.data[target_col] == 1][var]
                match_0 = self.data[self.data[target_col] == 0][var]

                if len(match_1) > 1 and len(match_0) > 1:
                    t_stat, p_value = stats.ttest_ind(match_1, match_0, nan_policy='omit')
                    print(f"{var}: t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}")

    def interest_correlation_analysis(self):
        """Analyze interest correlations and their impact"""
        if 'interests_correlate' in self.data.columns:
            print("=" * 50)
            print("INTERESTS CORRELATION ANALYSIS")
            print("=" * 50)

            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            # ensure numeric
            if not pd.api.types.is_numeric_dtype(self.data['interests_correlate']):
                self.data['interests_correlate'] = pd.to_numeric(self.data['interests_correlate'], errors='coerce')
            if self.data['interests_correlate'].dropna().empty:
                plt.text(0.5, 0.5, 'No numeric interests_correlate', horizontalalignment='center')
            else:
                self.data['interests_correlate'].hist(bins=20, alpha=0.7, color='orange')
            plt.title('Distribution of Interests Correlation')
            plt.xlabel('Correlation Coefficient')
            plt.ylabel('Frequency')

            if 'match' in self.data.columns:
                plt.subplot(1, 3, 2)
                try:
                    self.data.boxplot(column='interests_correlate', by='match', ax=plt.gca())
                except Exception:
                    pass
                plt.title('Interests Correlation by Match Status')
                plt.suptitle('')

                plt.subplot(1, 3, 3)
                try:
                    sns.scatterplot(data=self.data, x='interests_correlate', y='like', hue='match')
                except Exception:
                    pass
                plt.title('Interests Correlation vs Like Score')

            plt.tight_layout()
            plt.show()

            # Correlation statistics
            if 'match' in self.data.columns and pd.api.types.is_numeric_dtype(self.data['interests_correlate']):
                corr_with_match = self.data['interests_correlate'].corr(self.data['match'])
                print(f"Correlation between interests_correlate and match: {corr_with_match:.3f}")

    def decision_analysis(self):
        """Analyze decision-making process"""
        decision_cols = ['decision', 'decision_o', 'like', 'guess_prob_liked']
        available_decisions = [col for col in decision_cols if col in self.data.columns]

        if not available_decisions:
            return

        print("=" * 50)
        print("DECISION ANALYSIS")
        print("=" * 50)

        plt.figure(figsize=(15, 10))

        for i, col in enumerate(available_decisions[:4], 1):
            plt.subplot(2, 2, i)

            if pd.api.types.is_numeric_dtype(self.data[col]):
                self.data[col].hist(bins=20, alpha=0.7)
                plt.title(f'Distribution of {col}')
                plt.xlabel('Score')
                plt.ylabel('Frequency')
            else:
                self.data[col].value_counts().plot(kind='bar')
                plt.title(f'Distribution of {col}')
                plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

        # Decision vs actual match
        if 'decision' in self.data.columns and 'match' in self.data.columns:
            print("\nDecision vs Actual Match:")
            decision_match = pd.crosstab(self.data['decision'], self.data['match'],
                                         margins=True, margins_name="Total")
            print(decision_match)

    def comprehensive_eda(self, target_col='match'):
        """
        Run comprehensive EDA

        Parameters:
        target_col (str): Name of the target column
        """
        print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("=" * 60)

        self.basic_info()
        self.identify_column_types()
        self.target_variable_analysis(target_col)
        self.demographic_analysis()
        self.preference_analysis()
        self.correlation_analysis(target_col)
        self.match_success_analysis(target_col)
        self.interest_correlation_analysis()
        self.decision_analysis()

        print("=" * 60)
        print("EDA COMPLETED")
        print("=" * 60)


# Usage example:
def load_and_analyze_data(file_path, auto_clean=True, drop_threshold=0.5, outlier_z=4.0, verbose=False):
    """
    Load the speed dating data and run comprehensive EDA

    Parameters:
    file_path (str): Path to the CSV file
    auto_clean (bool): Whether to run automatic cleaning upon loading
    drop_threshold (float): Fraction of missing values above which a column is dropped
    outlier_z (float): z-score threshold used to optionally remove extreme outliers
    verbose (bool): If True, print more debug output from the cleaner
    """
    try:
        # Load the data
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully: {df.shape}")

        # Initialize and run EDA
        eda = SpeedDatingEDA(df, auto_clean=auto_clean, drop_threshold=drop_threshold, outlier_z=outlier_z, verbose=verbose)
        eda.comprehensive_eda()

        return df, eda

    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        return None, None


# Quick usage:
if __name__ == "__main__":
    df, eda = load_and_analyze_data('project/speeddating.csv', auto_clean=True, verbose=False)