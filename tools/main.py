import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class SpeedDatingEDA:
    def __init__(self, data):
        """
        Initialize the EDA class with the speed dating dataset

        Parameters:
        data (pd.DataFrame): The speed dating dataset
        """
        self.data = data.copy()
        self.numeric_columns = None
        self.categorical_columns = None

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
        self.categorical_columns = self.data.select_dtypes(include=['object']).columns.tolist()

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

        target_counts = self.data[target_col].value_counts()
        target_percentage = self.data[target_col].value_counts(normalize=True) * 100

        print("Target variable distribution:")
        for val, count, perc in zip(target_counts.index, target_counts.values, target_percentage.values):
            print(f"  {val}: {count} ({perc:.2f}%)")

        # Plot target distribution
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        target_counts.plot(kind='bar', color=['skyblue', 'lightcoral'])
        plt.title('Target Variable Distribution')
        plt.xlabel('Match')
        plt.ylabel('Count')
        plt.xticks(rotation=0)

        plt.subplot(1, 2, 2)
        plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
                colors=['lightblue', 'lightcoral'])
        plt.title('Match Percentage')

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
            gender_counts = self.data['gender'].value_counts()
            print(gender_counts)

            plt.figure(figsize=(15, 10))

            plt.subplot(2, 3, 1)
            gender_counts.plot(kind='bar', color=['lightpink', 'lightblue'])
            plt.title('Gender Distribution')
            plt.xticks(rotation=45)

        # Age analysis
        if 'age' in self.data.columns:
            print(f"\nAge Statistics:")
            print(f"Mean age: {self.data['age'].mean():.2f}")
            print(f"Median age: {self.data['age'].median():.2f}")
            print(f"Age range: {self.data['age'].min()} - {self.data['age'].max()}")

            plt.subplot(2, 3, 2)
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
                if self.data[col].dtype in [np.float64, np.int64]:
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

            if self.data[var].dtype in [np.float64, np.int64]:
                # For numeric variables, show distribution by match status
                self.data.boxplot(column=var, by=target_col, ax=plt.gca())
                plt.title(f'{var} by Match Status')
                plt.suptitle('')  # Remove automatic title
            else:
                # For categorical variables, show cross-tabulation
                cross_tab = pd.crosstab(self.data[var], self.data[target_col], normalize='index')
                cross_tab.plot(kind='bar', ax=plt.gca())
                plt.title(f'Match Rate by {var}')
                plt.xticks(rotation=45)

            plt.tight_layout()

        plt.show()

        # Statistical test for key variables
        print("\nStatistical significance of key variables with match success:")
        for var in available_vars[:5]:  # Test first 5 variables
            if self.data[var].dtype in [np.float64, np.int64]:
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
            self.data['interests_correlate'].hist(bins=20, alpha=0.7, color='orange')
            plt.title('Distribution of Interests Correlation')
            plt.xlabel('Correlation Coefficient')
            plt.ylabel('Frequency')

            if 'match' in self.data.columns:
                plt.subplot(1, 3, 2)
                self.data.boxplot(column='interests_correlate', by='match', ax=plt.gca())
                plt.title('Interests Correlation by Match Status')
                plt.suptitle('')

                plt.subplot(1, 3, 3)
                # Scatter plot with match success
                sns.scatterplot(data=self.data, x='interests_correlate', y='like', hue='match')
                plt.title('Interests Correlation vs Like Score')

            plt.tight_layout()
            plt.show()

            # Correlation statistics
            if 'match' in self.data.columns:
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

            if self.data[col].dtype in [np.float64, np.int64]:
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
def load_and_analyze_data(file_path):
    """
    Load the speed dating data and run comprehensive EDA

    Parameters:
    file_path (str): Path to the CSV file
    """
    try:
        # Load the data
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully: {df.shape}")

        # Initialize and run EDA
        eda = SpeedDatingEDA(df)
        eda.comprehensive_eda()

        return df, eda

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# Quick usage:
df, eda = load_and_analyze_data('project/speeddating.csv')