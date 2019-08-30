"""
Script containing utilities related to manual data cleaning, processing, and transformation.
"""

from pandas import read_csv, get_dummies, to_numeric
from pandas.api.types import is_numeric_dtype
import numpy as np
from scipy import stats


def unknown_to_nan(df, additional_regex_patterns=[], additional_strings=[]):
    """
    Function to convert typical unknown & unspecified values in a data frame to NaN values

    @param (pd.DataFrame) df: data frame with unknown values represented in various forms
    @param (list) additional_regex_patterns: regex patterns, alongside 'default_regex_patterns',
           that may be contained in unknown values in @df
    @param (list) additional_strings: strings, alongside 'default_strings', that may represent an
           unknown value in @df
    """
    default_regex_patterns = [r'^\s*$']  # blank character
    default_strings = ['n/a', 'N/A', 'NA', 'UNKNOWN', '-UNKNOWN-', '-unknown-', 'unknown']
    # Replace UNKNOWN tokens with proper missing value annotations
    for regex_pattern in default_regex_patterns + additional_regex_patterns:
        df.replace(to_replace=regex_pattern, value=np.nan, inplace=True, regex=True)
    for string in default_strings + additional_strings:
        df.replace(to_replace=string, value=np.nan, inplace=True)
    return df


def categorical_to_onehot_columns(df, target_column=None):
    """
    Function to encode categorical columns (features) into one-hot encoded columns.

    @param (pd.DataFrame) df: data frame with categorical variables as columns
    @param (str) target_column: column name for the dependent target variable in @df (default: None)
    """
    for column in df.columns.values:
        if not is_numeric_dtype(df[column]) and not column == target_column:
            # Convert to one-hot representation
            onehot_column = get_dummies(data=df[column], prefix=column, prefix_sep='=')
            # Replace all 0 rows (deriving from NaN values) with the mean of the one-hot column
            onehot_column.loc[(onehot_column == 0).all(axis=1)] = onehot_column.mean().tolist()
            # Join with the original data frame and remove the original categorical column
            df = df.join(other=onehot_column).drop(column, axis=1)
    return df


def groupby_and_gather(raw_df):
    """
    Function to groupby by age-gender buckets and gather the population per country, included as
    a problem-specific example rather than a generalized utility.

    @param (pd.DataFrame) raw_df: data frame with age-gender buckets and corresponding information
    """
    raw_df['bucket_id'] = raw_df['age_bucket'].str.replace('+', '-200') + raw_df['gender']
    raw_df.drop(['year', 'age_bucket', 'gender'], axis=1, inplace=True)
    for column in set(raw_df['country_destination'].values):
        raw_df[column] = 0

    for country in set(raw_df['country_destination'].values.tolist()):
        raw_df.loc[raw_df['country_destination'] == country, country] = 1

    raw_df.drop('country_destination', axis=1, inplace=True)

    for column in raw_df.columns.values:
        if column != 'bucket_id' and column != 'population_in_thousands':
            raw_df[column] = raw_df[column] * raw_df['population_in_thousands']

    sum_df = raw_df.groupby(['bucket_id']).sum()
    sum_df.drop('population_in_thousands', axis=1, inplace=True)
    sum_df.reset_index(inplace=True)
    return sum_df


string_age_buckets = set(
    read_csv('(0)data/age_gender_bkts.csv')['age_bucket'].str.replace('+', '-200').values.tolist()
)
range_to_string_mapping = {tuple([i for i in range(int(rep.split('-')[0]),
                                                   int(rep.split('-')[1]) + 1)]): rep
                           for rep in string_age_buckets}


def age_to_age_bucket(age):
    """Function to convert a given integer age value to the corresponding age bucket"""
    for bucket in range_to_string_mapping.keys():
        if int(age) in tuple(bucket):
            return range_to_string_mapping[bucket]


def mcar_test(df, significance_level=0.05):
    """
    Function for performing Little's chi-square test (1988) for the assumption (null hypothesis) of
    missing completely at random (MCAR). Data should be multivariate and quantitative, categorical
    variables do not work. The null hypothesis is equivalent to saying that the missingness of the
    data is independent of both the observed and unobserved data. Common imputation methods like
    likelihood inference and listwise deletion are theorized to be valid (due to non-inclusion of
    bias) only when the data is missing at random (MAR), which MCAR is a subset of.

    @param (pd.DataFrame) df: data frame with missing values that are ideally all quantitative
    @param (float) significance_level: alpha parameter of the chi-squared test (default: 0.05)
    """
    test_df = df.copy()
    # Check if data contains categorical variables and select only numerical (float) columns
    if False in [is_numeric_dtype(test_df[column]) for column in test_df.columns.values]:
        test_df = test_df.select_dtypes([np.number]).apply(to_numeric)

    # Estimate means and covariances
    # TODO: It would be better if we used unbiased estimators here as well
    means, covariances = test_df.mean(), test_df.cov()

    # Get missing data patterns
    all_patterns = 1 * test_df.isnull().to_numpy()
    unique_patterns = np.unique(all_patterns, axis=0).tolist()
    # Get pattern-to-id mapping for matching data frame rows and missing data patterns
    pattern_to_id = dict(zip([tuple(pattern) for pattern in unique_patterns],
                             [i for i in range(len(unique_patterns))]))
    # Add patterns as a column for identification
    test_df['pattern_id'] = [pattern_to_id[tuple(pattern)] for pattern in all_patterns.tolist()]
    # Initialize statistic variables
    test_statistic, total_observed_variables = 0, 0

    for pattern in unique_patterns:
        # Get samples with the current pattern
        sample = test_df[test_df['pattern_id'] == pattern_to_id[tuple(pattern)]]
        # Drop unrelated 'pattern_id' column
        sample = sample.drop('pattern_id', axis=1)
        # Get observed variables
        observed_variables = sample.columns[~sample.isnull().any()].tolist()
        total_observed_variables += len(observed_variables)
        # Measure the sample mean & global mean & related (based on observed variables) covariance
        sample_mean = sample[observed_variables].mean().to_numpy()
        global_mean = means[observed_variables].to_numpy()
        related_covariance = covariances.reindex(index=observed_variables,
                                                 columns=observed_variables).to_numpy()
        # Get test-statistic portion from the current pattern
        test_statistic += len(sample) * np.dot(
            a=(sample_mean - global_mean).T,
            b=np.dot(np.linalg.inv(related_covariance), sample_mean - global_mean)
        )

    # Estimate p-value from chi-squared test
    p_value = 1 - stats.chi2.cdf(x=test_statistic,
                                 df=total_observed_variables - len(test_df.columns))
    # NOTE: Don't confuse the degrees of freedom (df) parameter with data frame (@df) argument

    # Fail to reject the null hypothesis, data is assumed to be MCAR
    if p_value > significance_level:
        return True
    # Reject the null hypothesis, data is not MCAR (it could be MAR or MNAR)
    else:
        return False
