import sys
sys.path.append("../../src")

import pandas as pd
import numpy as np

import video_analyses


################### EEG DATA PROCESSING ###################

# def read_and_preprocess_csv(file_path):
#     df = pd.read_csv(file_path)
#     df.columns = df.iloc[0]
#     df = df.iloc[1:]
#     df['Time'] = np.arange(1, len(df) * 1 + 1, 1)
#     return df

# def calculate_total_power(alpha_df, beta_df, theta_df, lgamma_df):
#     total_power_df = pd.DataFrame()
#     for column in alpha_df:
#         if column != 'Time':
#             total_power_df[column] = (
#                 alpha_df[column].astype(float) + 
#                 beta_df[column].astype(float) + 
#                 theta_df[column].astype(float) + 
#                 lgamma_df[column].astype(float)
#             )
#     return total_power_df

# def calculate_relative_power(band_df, total_power_df):
#     relative_power_df = pd.DataFrame()
#     for column in band_df:
#         if column != 'Time':
#             relative_power_df[column] = band_df[column].astype(float) / total_power_df[column].astype(float)
#     relative_power_df['Time'] = np.arange(1, len(relative_power_df) * 1 + 1, 1)
#     return relative_power_df

# def calculate_alpha_theta_ratio(alpha_relative, theta_relative):
#     alpha_theta_ratio_df = pd.DataFrame()
#     for column in alpha_relative:
#         if column != 'Time':
#             alpha_theta_ratio_df[column] = alpha_relative[column].astype(float) / theta_relative[column].astype(float)
#     alpha_theta_ratio_df['Time'] = np.arange(1, len(alpha_theta_ratio_df) * 1 + 1, 1)
#     return alpha_theta_ratio_df

# def calculate_averages_by_region(input_df, regions, bandname):

#     averages_by_region = {}

#     for electrode in input_df.columns:
#         input_df[electrode] = pd.to_numeric(input_df[electrode], errors='coerce')

#     # Loop through the regions and calculate the mean for each region
#     for region, electrodes in regions.items():
#         # Filter the electrodes that exist in the DataFrame
#         valid_electrodes = [elec for elec in electrodes if elec in input_df.columns and elec != 'Time']
        
#         if valid_electrodes:
#             region_data = input_df[valid_electrodes]
#             average_data = region_data.mean(axis=1)  # Calculate the mean across columns (electrodes)
#             averages_by_region[region] = average_data

#     averages_df = pd.DataFrame(averages_by_region)
#     averages_df.columns = [col + '_' + bandname for col in averages_df.columns]
#     averages_df['Time'] = input_df['Time']

#     return averages_df

################### EEG RETRIEVAL AND PROCESSING ###################

# def process_participant_data(participants, address_CENIR, regions):
#     # Initialize dictionaries to store results
#     alpha_averages_dict = {}
#     beta_averages_dict = {}
#     theta_averages_dict = {}
#     lgamma_averages_dict = {}
#     alpha_theta_ratio_avg_dict = {}

#     # Iterate over each participant
#     for i in participants:
#         # Fetch the .csv files of all participants
#         sub_id = str(i).zfill(3)  # Fill with putting 0 in front of the number
#         alpha_file = address_CENIR + 'Results_PSD/PSD_time_3D_Sub' + sub_id + '_Alpha.csv'
#         beta_file = address_CENIR + 'Results_PSD/PSD_time_3D_Sub' + sub_id + '_Beta.csv'
#         theta_file = address_CENIR + 'Results_PSD/PSD_time_3D_Sub' + sub_id + '_Theta.csv'
#         lgamma_file = address_CENIR + 'Results_PSD/PSD_time_3D_Sub' + sub_id + '_Lgamma.csv'
        
#         # Read and preprocess the data
#         EEG_df_sub_alpha = read_and_preprocess_csv(alpha_file)
#         EEG_df_sub_beta = read_and_preprocess_csv(beta_file)
#         EEG_df_sub_theta = read_and_preprocess_csv(theta_file)
#         EEG_df_sub_lgamma = read_and_preprocess_csv(lgamma_file)

#         # Calculate total power
#         EEG_df_sub_total_power = calculate_total_power(EEG_df_sub_alpha, EEG_df_sub_beta, EEG_df_sub_theta, EEG_df_sub_lgamma)
        
#         # Calculate relative power for each band
#         EEG_df_sub_alpha_relative_power = calculate_relative_power(EEG_df_sub_alpha, EEG_df_sub_total_power)
#         EEG_df_sub_beta_relative_power = calculate_relative_power(EEG_df_sub_beta, EEG_df_sub_total_power)
#         EEG_df_sub_theta_relative_power = calculate_relative_power(EEG_df_sub_theta, EEG_df_sub_total_power)
#         EEG_df_sub_lgamma_relative_power = calculate_relative_power(EEG_df_sub_lgamma, EEG_df_sub_total_power)

#         # Calculate alpha / theta ratio
#         EEG_df_sub_alpha_theta_ratio = calculate_alpha_theta_ratio(EEG_df_sub_alpha_relative_power, EEG_df_sub_theta_relative_power)

#         # Calculate averages by region
#         alpha_averages_df = calculate_averages_by_region(EEG_df_sub_alpha_relative_power, regions, bandname='alpha')
#         beta_averages_df = calculate_averages_by_region(EEG_df_sub_beta_relative_power, regions, bandname='beta')
#         theta_averages_df = calculate_averages_by_region(EEG_df_sub_theta_relative_power, regions, bandname='theta')
#         lgamma_averages_df = calculate_averages_by_region(EEG_df_sub_lgamma_relative_power, regions, bandname='lgamma')
#         alpha_theta_ratio_avg_df = calculate_averages_by_region(EEG_df_sub_alpha_theta_ratio, regions, bandname='alpha_theta_ratio')

#         # Remove leading zeros from sub_id for the dictionary key
#         sub_id = str(i).zfill(2)

#         # Store the averages in the corresponding dictionaries with sub_id as key
#         alpha_averages_dict[sub_id] = alpha_averages_df
#         beta_averages_dict[sub_id] = beta_averages_df
#         theta_averages_dict[sub_id] = theta_averages_df
#         lgamma_averages_dict[sub_id] = lgamma_averages_df
#         alpha_theta_ratio_avg_dict[sub_id] = alpha_theta_ratio_avg_df

#     # Return the dictionaries
#     return {
#         'alpha_averages': alpha_averages_dict,
#         'beta_averages': beta_averages_dict,
#         'theta_averages': theta_averages_dict,
#         'lgamma_averages': lgamma_averages_dict,
#         'alpha_theta_ratio_avg': alpha_theta_ratio_avg_dict
#     }


def calculate_average_by_ID_all(alpha_averages, alpha_averages_2D, beta_averages, beta_averages_2D, segmentation, relaxed_IDs):
    # ALPHA
    avg_by_ID_alpha_3D = calculate_average_by_ID(alpha_averages, 'ID', ['Time'], segmentation) 
    avg_by_ID_alpha_2D = calculate_average_by_ID(alpha_averages_2D, 'ID', ['Time'], segmentation)

    avg_by_ID_alpha_3D['condition'] = 0
    avg_by_ID_alpha_3D = avg_by_ID_alpha_3D.reset_index()
    avg_by_ID_alpha_3D['group'] = np.where(avg_by_ID_alpha_3D['ID'].isin(relaxed_IDs), 0, 1)

    avg_by_ID_alpha_2D['condition'] = 1
    avg_by_ID_alpha_2D = avg_by_ID_alpha_2D.reset_index()
    avg_by_ID_alpha_2D['group'] = np.where(avg_by_ID_alpha_2D['ID'].isin(relaxed_IDs), 0, 1)

    avg_by_ID_alpha_regions = pd.concat([avg_by_ID_alpha_3D, avg_by_ID_alpha_2D], axis=0)

    # BETA
    avg_by_ID_beta_3D = calculate_average_by_ID(beta_averages, 'ID', ['Time'], segmentation)
    avg_by_ID_beta_2D = calculate_average_by_ID(beta_averages_2D, 'ID', ['Time'], segmentation)

    avg_by_ID_beta_3D['condition'] = 0
    avg_by_ID_beta_3D = avg_by_ID_beta_3D.reset_index()
    avg_by_ID_beta_3D['group'] = np.where(avg_by_ID_beta_3D['ID'].isin(relaxed_IDs), 0, 1)

    avg_by_ID_beta_2D['condition'] = 1
    avg_by_ID_beta_2D = avg_by_ID_beta_2D.reset_index()
    avg_by_ID_beta_2D['group'] = np.where(avg_by_ID_beta_2D['ID'].isin(relaxed_IDs), 0, 1)

    avg_by_ID_beta_regions = pd.concat([avg_by_ID_beta_3D, avg_by_ID_beta_2D], axis=0)

    # MELTING
    melted_data_EEG_alpha_regions = pd.melt(avg_by_ID_alpha_regions, id_vars=['ID', 'condition', 'group'], var_name='variable', value_name='value')
    melted_data_EEG_beta_regions = pd.melt(avg_by_ID_beta_regions, id_vars=['ID', 'condition', 'group'], var_name='variable', value_name='value')

    melted_data_EEG_alpha_regions['ID'] = melted_data_EEG_alpha_regions['ID'].astype(int)
    melted_data_EEG_beta_regions['ID'] = melted_data_EEG_beta_regions['ID'].astype(int)

    return melted_data_EEG_alpha_regions, melted_data_EEG_beta_regions

def calculate_average_by_ID(data, group_col, value_cols, segmentation):
    """
    Calculate the average values for each ID in a DataFrame.

    Parameters:
        data (DataFrame): The input DataFrame.
        group_col (str): The column to group by (e.g., 'ID').
        value_cols (list): A list of columns to calculate the mean for.
        ignore_breathing (boolean): if 1, computes the average on starting after the end of breathing exercise 

    Returns:
        DataFrame: A DataFrame containing the average values for each ID.
    """

    if segmentation == 'breathing':
        data = data[data['Time'] <= 188] # the breathing ends at 188 seconds
    elif segmentation == 'after':
        data = data[data['Time'] > 188] # the breathing ends at 188 seconds

    grouped_data = data.groupby(group_col)
    mean_values = grouped_data.mean()
    avg_by_ID = pd.DataFrame(mean_values)
    avg_by_ID = avg_by_ID.drop(columns=value_cols)
    return avg_by_ID

################
# CALLED JUST BEFORE DOING EEG TEMPORAL PLOTS

# Computes the average across participants of same group (all, resp, unresp) 
def process_EEG_by_group(alpha_averages, beta_averages, alpha_averages_2D, beta_averages_2D, IDs):
    # select the right participants based on IDs, and then calculate the avg and std for each freq. band (alpha, beta) and the chosen region (midline)

    regions_alpha = calculate_grouped_by_region_statistics(alpha_averages, 'alpha', IDs)
    regions_beta = calculate_grouped_by_region_statistics(beta_averages, 'beta', IDs)
    regions_2D_alpha = calculate_grouped_by_region_statistics(alpha_averages_2D, 'alpha', IDs)
    regions_2D_beta = calculate_grouped_by_region_statistics(beta_averages_2D, 'beta', IDs)

    regions_alpha['Avg'] = regions_alpha['midline_alpha_mean']
    regions_alpha['Std'] = regions_alpha['midline_alpha_std']
    regions_alpha['time'] = regions_alpha['Time']

    regions_beta['Avg'] = regions_beta['midline_beta_mean']
    regions_beta['Std'] = regions_beta['midline_beta_std']
    regions_beta['time'] = regions_beta['Time']

    regions_2D_alpha['Avg'] = regions_2D_alpha['midline_alpha_mean']
    regions_2D_alpha['Std'] = regions_2D_alpha['midline_alpha_std']
    regions_2D_alpha['time'] = regions_2D_alpha['Time']

    regions_2D_beta['Avg'] = regions_2D_beta['midline_beta_mean']
    regions_2D_beta['Std'] = regions_2D_beta['midline_beta_std']
    regions_2D_beta['time'] = regions_2D_beta['Time']

    return regions_alpha, regions_beta, regions_2D_alpha, regions_2D_beta

def calculate_grouped_by_region_statistics(dataframe, bandname, IDs):
    # Called by process_EEG_by_group, computes avg and std for the selected band (alpha, beta), for each region

    # PAUL : only keep the participants from the IDs list (all, unresponsive, relaxed)
    dataframe_ID = dataframe[dataframe["ID"].str.contains("|".join(IDs))]

    grouped_data = dataframe_ID.groupby("Time")

    # Calculate mean and standard deviation for each region within each time point
    grouped_data = grouped_data.agg({f"frontal_{bandname}": ["mean", "std"],
                                     f"temporal_{bandname}": ["mean", "std"],
                                     f"central_{bandname}": ["mean", "std"],
                                     f"parietal_{bandname}": ["mean", "std"],
                                     f"occipital_{bandname}": ["mean", "std"],
                                     f"midline_{bandname}": ["mean", "std"]})

    grouped_data.columns = [f"{region}_{stat}" for region, stat in grouped_data.columns]

    grouped_data.reset_index(inplace=True)

    return grouped_data


################### ECG PROCESSING ###################

def calculate_average_by_ID_all_ECG(data3D, data2D):

    data3D['condition'] = 0
    data2D['condition'] = 1
    combined_data = pd.concat([data3D, data2D], axis=0)

    combined_data = combined_data[combined_data['time'] > 220]

    if combined_data.columns.str.contains('LF').any():
        combined_data.drop(columns = ['Std_smooth','Avg_smooth'], inplace=True)

    # TO EXCLUDE P25 IF STILL PRESENT IN DATA
    combined_data.drop(columns= [col for col in combined_data.columns if '25' in col], inplace=True)

    combined_data.drop(columns = ['Unnamed: 0','Std','Avg','time'], inplace=True)

    # # STANDARDISATION WITH BOTH 2D AND 3D DATA
    data_values = combined_data.drop(columns=['condition']).values
    data_standardized_values = (data_values - np.nanmean(data_values)) / np.nanstd(data_values)
    data_standardized = pd.DataFrame(data_standardized_values, columns=combined_data.drop(columns=['condition']).columns)
    data_standardized['condition'] = combined_data['condition'].values

    data_mean = data_standardized.groupby('condition').mean().reset_index()
    data_melted = data_mean.melt(id_vars=['condition'], var_name='variable', value_name='value')
    data_melted['ID'] = data_melted['variable'].str.extract(r'(\d+)').astype(int)
    data_melted['variable'] = data_melted['variable'].str.extract(r'([a-zA-Z_]+)')
    data_melted['variable'] = data_melted['variable'].str.rstrip('_') # remove the last _

    return data_melted[['ID','condition','variable','value']].sort_values(by=['condition', 'ID'])

############

# CALLED BEFORE DOING THE ECG TEMPORAL PLOTS : 
def divide_ECG_into_groups(df_3D, df_2D, subgroups, ECG_param):
    """divide ECG dfs into unresponsive and relaxed groups

    Parameters
    ----------
    df_3D : dataframe
        zen garden df
    df_2D : dataframe
        city df
    subgroups : list[list[str]]
        subgroups listed as unresponsive and relaxed, defined in the function
    ECG_param : str
        ECG parameter to look at (LF, HF, Fratio)

    Returns
    -------
    unresponsive_3D_df : dataframe
    unresponsive_2D_df : dataframe
    relaxed_3D_df : dataframe
    relaxed_2D_df : dataframe
    """

    # unresponsives 3D
    unresp_df_3D = [col for col in df_3D.columns if any(id_ in col for id_ in subgroups[0])]
    unresponsive_df_3D = df_3D[unresp_df_3D]

    # unresponsives 2D
    unresp_df_2D = [col for col in df_2D.columns if any(id_ in col for id_ in subgroups[0])]
    unresponsive_df_2D = df_2D[unresp_df_2D]

    # relaxed 3D
    relx_df_3D = [col for col in df_3D.columns if any(id_ in col for id_ in subgroups[1])]
    relaxed_df_3D = df_3D[relx_df_3D] 

    # relaxed 2D
    relx_df_2D = [col for col in df_2D.columns if any(id_ in col for id_ in subgroups[1])]
    relaxed_df_2D = df_2D[relx_df_2D]   

    # Add average and standard deviation columns
    unresponsive_df_3D = video_analyses.add_avg_std_time_columns(unresponsive_df_3D, ECG_param)
    unresponsive_df_2D = video_analyses.add_avg_std_time_columns(unresponsive_df_2D, ECG_param)
    relaxed_df_3D = video_analyses.add_avg_std_time_columns(relaxed_df_3D, ECG_param)
    relaxed_df_2D = video_analyses.add_avg_std_time_columns(relaxed_df_2D, ECG_param)

    return unresponsive_df_3D, unresponsive_df_2D, relaxed_df_3D, relaxed_df_2D


################### OTHER PROCESSING ###################

def means_by_region(df, group_option='all'):
    """
    Calculate the mean of the 'value' column in the DataFrame based on the specified group_option.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame with columns ['ID', 'condition', 'group', 'variable', 'value'].
    group_option (str): Specify whether to calculate the mean for:
                        - 'all' (all groups combined)
                        - 'groups' (separately for group 0 and group 1, concatenated in the same DataFrame)
    
    Returns:
    pd.DataFrame: The resulting DataFrame with calculated mean values.
    """
    
    # Check if the group_option is valid
    if group_option not in ['all', 'groups']:
        raise ValueError("Invalid group_option. Choose from 'all' or 'groups'.")
    
    # Case 1: Calculate mean for all groups combined (ignoring the 'group' column)
    if group_option == 'all':
        result_df = df.groupby(['condition', 'variable'])['value'].mean().reset_index()
    
    # Case 2: Calculate mean separately for group 0 and group 1, then concatenate the results
    elif group_option == 'groups':
        # Calculate the mean for group 0
        group_0_df = df[df['group'] == 0].groupby(['condition', 'variable', 'group'])['value'].mean().reset_index()
        
        # Calculate the mean for group 1
        group_1_df = df[df['group'] == 1].groupby(['condition', 'variable', 'group'])['value'].mean().reset_index()
        
        # Concatenate both results
        result_df = pd.concat([group_0_df, group_1_df]).reset_index(drop=True)
    
    return result_df


def melt_EEG_data(df, bandname, id_vars, calculate_mean=False):
    """
    Generalized function to melt EEG data and optionally calculate means.

    Args:
        df (pd.DataFrame): The input DataFrame.
        bandname (str): The bandname to dynamically generate columns for melting.
        id_vars (list): List of columns to retain as identifiers during melting.
        calculate_mean (bool): Whether to calculate means after melting.

    Returns:
        pd.DataFrame: The melted DataFrame, optionally with means calculated.
    """
    # Dynamically generate the columns to melt based on the bandname
    columns_to_melt = [
        f"frontal_{bandname}",
        f"temporal_{bandname}",
        f"central_{bandname}",
        f"parietal_{bandname}",
        f"occipital_{bandname}",
        f"midline_{bandname}"
    ]

    # Melt the DataFrame
    melted = df.melt(
        id_vars=id_vars,         # Identifier columns
        value_vars=columns_to_melt,  # Columns to melt
        var_name='variable',     # Name for the new "variable" column
        value_name='value'       # Name for the new "value" column
    )

    if calculate_mean:
        # Calculate means based on id_vars + 'variable'
        melted = (
            melted.groupby(id_vars + ['variable'], as_index=False)['value']
            .mean()
            .sort_values(by=id_vars + ['variable'])  # Ensure consistent sorting
        )
        # Round to six decimal places for readability
        melted['value'] = melted['value'].round(6)

    return melted

def process_EEG_data_violin(df, bandname):
    # Specific implementation for the first function
    return melt_EEG_data(df, bandname, id_vars=['ID', 'condition', 'group'], calculate_mean=False)

def process_EEG_for_topoplot(df, bandname):
    # Specific implementation for the second function
    return melt_EEG_data(df, bandname, id_vars=['condition', 'group'], calculate_mean=True)



######## CALCULATE CONTRASTS BETWEEN CONDITIONS FOR TOPOMAP PLOTS ########

def extract_contrast_values(df, group_number, bandname):
    """
    Extract contrast values for specified group and band name from a DataFrame.

    Parameters:
    - df: DataFrame containing beta or high alpha contrasts.
    - group_number: The group number (0 or 1) to filter the DataFrame.
    - bandname: The band name ('beta' or 'high alpha') to adjust variable names.

    Returns:
    - A DataFrame with contrast values for different regions.
    """
    # Ensure bandname is valid
    if bandname not in ['beta', 'high_alpha']:
        raise ValueError("Invalid band name. Must be 'beta' or 'high_alpha'.")

    # Filter the DataFrame based on the group number
    contrasts_group = df[df['group'] == group_number].copy(deep=True)
    
    # Pivot the DataFrame
    pivotdf = contrasts_group.pivot(index='variable', columns='condition', values='value').reset_index()
    
    # Calculate the contrast (condition 0 - condition 1)
    if 0 in pivotdf.columns and 1 in pivotdf.columns:
        pivotdf['contrast'] = pivotdf[0] - pivotdf[1]
    else:
        raise KeyError("Columns 0 or 1 are missing from the DataFrame.")
    
    # Adjust variable names based on band name
    variable_names = {
        'central_value': f'central_{bandname}',
        'frontal_value': f'frontal_{bandname}',
        'occipital_value': f'occipital_{bandname}',
        'parietal_value': f'parietal_{bandname}',
        'temporal_value': f'temporal_{bandname}',
        'midline_value': f'midline_{bandname}'
    }
    
    # Convert the DataFrame to a dictionary
    values = pivotdf.set_index('variable')['contrast'].to_dict()
    
    # Extract specific values, defaulting to None if not found
    extracted_values = {
        region: values.get(var_name, None)
        for region, var_name in variable_names.items()
    }
    
    # Convert the extracted values into a DataFrame
    extracted_df = pd.DataFrame([extracted_values])

    # Return the extracted DataFrame
    return extracted_df


################### remove outlier brain-heart calculation ###################

def remove_outliers(df, variable_col='variable', ID_column='ID', value_col='value'):
    def is_outlier(s):
        Q1 = s.quantile(0.05)
        Q3 = s.quantile(0.95)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ~((s >= lower_bound) & (s <= upper_bound))

    # Apply outlier detection per variable group
    df['is_outlier'] = df.groupby(variable_col)[value_col].transform(is_outlier)

    # Print rows that are actual outliers
    outliers = df[df['is_outlier'] == True]
    if not outliers.empty:
        print("Aberrant data detected (outliers):")
        print(outliers)

    # Identify IDs associated with outliers
    outlier_ids = df.loc[df['is_outlier'] == True, ID_column].unique()

    # Remove rows where the ID matches any outlier ID
    df_cleaned = df[~df[ID_column].isin(outlier_ids)].copy()

    # Drop the helper column
    df_cleaned.drop(columns=['is_outlier'], inplace=True)

    return df_cleaned