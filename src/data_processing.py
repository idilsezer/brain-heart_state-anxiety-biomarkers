import sys
sys.path.append("../../src")

import pandas as pd
import numpy as np

import video_analyses

# import warnings
# warnings.filterwarnings("ignore")


################### RETRIEVE EEG DATA ###################

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





################### EEG PROCESSING ###################


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





################### TOPOPLOTS ###################


def extract_contrast_values(df, group_number, band_name):
    """
    Extract contrast values for specified group and band name from a DataFrame.

    Parameters:
    - df: DataFrame containing beta or alpha contrasts.
    - group_number: The group number (0 or 1) to filter the DataFrame.
    - band_name: The band name ('beta' or 'alpha') to adjust variable names.

    Returns:
    - A dictionary with contrast values for different regions.
    """
    # Ensure band_name is valid
    if band_name not in ['beta', 'alpha']:
        raise ValueError("Invalid band_name. Must be 'beta' or 'alpha'.")

    # Filter the DataFrame based on the group number
    contrasts_group = df[df['group'] == group_number].copy(deep=True)
    
    # Pivot the DataFrame
    pivotdf = contrasts_group.pivot(index='variable', columns='condition', values='value').reset_index()
    
    # Calculate the contrast (condition 0 - condition 1)
    if 0 in pivotdf.columns and 1 in pivotdf.columns:
        pivotdf['contrast'] = pivotdf[0] - pivotdf[1]
    else:
        raise KeyError("Columns 0 or 1 are missing from the DataFrame.")
    
    # Adjust variable names based on band_name
    variable_names = {
        'central_value': f'central_{band_name}',
        'frontal_value': f'frontal_{band_name}',
        'occipital_value': f'occipital_{band_name}',
        'parietal_value': f'parietal_{band_name}',
        'temporal_value': f'temporal_{band_name}'
    }
    
    # Convert the DataFrame to a dictionary
    values = pivotdf.set_index('variable')['contrast'].to_dict()
    
    # Extract specific values, defaulting to None if not found
    extracted_values = {
        region: values.get(var_name, None)
        for region, var_name in variable_names.items()
    }
    
    # Return the extracted values
    return extracted_values
