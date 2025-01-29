import sys
import os
sys.path.append("../../src")
import pandas as pd
# import flirt.reader.empatica 
# from datetime import datetime, timedelta, date
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
# import emotion
import numpy as np
import seaborn as sns


def extract_column_from_dict(video_dict, column_of_interest, prefix=''):
    """function to extract a column of interest from a dictionary of df

    Parameters
    ----------
    video_dict : dict
        dictionary of all participants' df
    column_of_interest : str
        name of the column of interest (e.g. 'LF_avg', 'HF_avg', 'LF_HF_ratio'...)
    prefix : str, optional
        prefix to add to the column of interest (e.g. 'zen_garden', 'neutral_city'...)
    Returns
    -------
    df
        a single df with the column of interest for all participants
    """
    result_df = pd.DataFrame()

    for participant_id, df in video_dict.items():
        new_column_name = f'{prefix}_{column_of_interest}_{participant_id}' if prefix else f'{column_of_interest}_{participant_id}'
        result_df[new_column_name] = df[column_of_interest]

    return result_df


def add_avg_std_time_columns(df, column_of_interest, time=True):
    """generate Avg Std and time columns for a given df

    Parameters
    ----------
    df : dataframe
        df of all participants' single type of data
    column_of_interest : str
        name of the column of interest (e.g. 'LF_avg', 'HF_avg', 'LF_HF_ratio'...)

    Returns
    -------
    df 
        df with added Avg, Std, and time columns
    """
    df['Avg'] = df.filter(like=column_of_interest).mean(axis=1)
    df['Std'] = df.filter(like=column_of_interest).std(axis=1)
    if time == True:
        df['time'] = range(60, 60 + len(df) * 5, 5)
    return df

import pandas as pd
import matplotlib.pyplot as plt

# def plot_comparison(df1, df2, column_of_interest, ymin_value=18, ymax_value=125, highlight_epoch_start=128, highlight_epoch_end=225):
#     """plot to compare the average of a column of interest between zen garden and neutral city videos

#     Parameters
#     ----------
#     df1 : dataframe
#         zen garden dataframe
#     df2 : dataframe
#         neutral city dataframe
#     column_of_interest : str
#         name of the column of interest (e.g. this time 'LF', 'HF', 'LF_HF_ratio'...)
#     ymin_value : int, optional
#         min range of breathing epoch, by default 18
#     ymax_value : int, optional
#         max range of breathing epoch, by default 125
#     highlight_epoch_start : int, optional
#         start of breathin epoch, by default 128
#     highlight_epoch_end : int, optional
#         end of breathing epoch, by default 225
#     """
#     plt.style.use('seaborn-white')
#     # Calculate rolling mean for 'Std' in the first DataFrame
#     df1['Std_smooth'] = df1['Std'].rolling(window=5, min_periods=1).mean()

#     # Calculate rolling mean for 'Std' in the second DataFrame
#     df2['Std_smooth'] = df2['Std'].rolling(window=5, min_periods=1).mean()

#     # Create a figure and axis
#     fig, ax = plt.subplots(figsize=(20, 10))

#     # Plot the 'Avg' line for the first DataFrame
#     ax.plot(df1['time'], df1['Avg'], label=f'Avg - zen garden', linewidth=2, color='blue')

#     # Fill the area between 'Avg' and smoothed 'Std' for the first DataFrame with a shaded region
#     ax.fill_between(df1['time'], df1['Avg'] - df1['Std_smooth'],
#                     df1['Avg'] + df1['Std_smooth'], color='blue', alpha=0.05, label=f'Std - zen garden')

#     # Plot the 'Avg' line for the second DataFrame
#     ax.plot(df2['time'], df2['Avg'], label=f'Avg - neutral city', linewidth=2, color='red')

#     # Fill the area between 'Avg' and smoothed 'Std' for the second DataFrame with a shaded region
#     ax.fill_between(df2['time'], df2['Avg'] - df2['Std_smooth'],
#                     df2['Avg'] + df2['Std_smooth'], color='red', alpha=0.05, label=f'Std - neutral city')

#     # Highlight the epoch on the 'Avg - Zen Garden' plot with a shaded region
#     ymin_norm = (ymin_value - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
#     ymax_norm = (ymax_value - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])

#     ax.axvspan(xmin=highlight_epoch_start, xmax=highlight_epoch_end, ymin=ymin_norm, ymax=ymax_norm,
#                color='green', alpha=0.2, label='Breathing epoch during zen garden')

#     ax.set_xlabel("Time (seconds)", size=16)
#     ax.set_ylabel(f"{column_of_interest} Value", size=16)

#     # Set y-axis limits to a larger range if needed
#     # ax.set_ylim(bottom=-50, top=150)

#     # Place the legend in a separate box on the right
#     ax.legend(loc='upper right')

#     ax.set_title(f"{column_of_interest} average comparison between zen garden and neutral city videos", size=20)

#     plt.show()

import matplotlib.pyplot as plt

def plot_comparison(df1, df2, column_of_interest, ymin_value=18, ymax_value=125, highlight_epoch_start=128, highlight_epoch_end=225, smoothness=2):
    """plot to compare the average of a column of interest between zen garden and neutral city videos

    Parameters
    ----------
    df1 : dataframe
        zen garden dataframe
    df2 : dataframe
        neutral city dataframe
    column_of_interest : str
        name of the column of interest (e.g. this time 'LF', 'HF', 'LF_HF_ratio'...)
    ymin_value : int, optional
        min range of breathing epoch, by default 18
    ymax_value : int, optional
        max range of breathing epoch, by default 125
    highlight_epoch_start : int, optional
        start of breathin epoch, by default 128
    highlight_epoch_end : int, optional
        end of breathing epoch, by default 225
    smoothness : int, optional
        window size for smoothing, by default 2
    """
    plt.style.use('seaborn-white')
    # Calculate rolling mean for 'Std' in the first DataFrame
    df1['Std_smooth'] = df1['Std'].rolling(window=smoothness, min_periods=1).mean() # window=2 for ECG; =10 for EEG

    # Calculate rolling mean for 'Std' in the second DataFrame
    df2['Std_smooth'] = df2['Std'].rolling(window=smoothness, min_periods=1).mean()

    # Calculate rolling mean for 'Avg' in the first DataFrame
    df1['Avg_smooth'] = df1['Avg'].rolling(window=smoothness, min_periods=1).mean() #set to window=2 for LF; 5 for HF; 2 for LF_HF_ratio: 10 for EEG

    # Calculate rolling mean for 'Avg' in the second DataFrame
    df2['Avg_smooth'] = df2['Avg'].rolling(window=smoothness, min_periods=1).mean()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(20, 10)) #changed from (20, 10) to (20, 15)

    # Plot the smoothed 'Avg' line for the first DataFrame
    ax.plot(df1['time'], df1['Avg_smooth'], label=f'Avg - zen garden (smoothed)', linewidth=1.5, color='blue')

    # Fill the area between smoothed 'Avg' and smoothed 'Std' for the first DataFrame with a shaded region
    ax.fill_between(df1['time'], df1['Avg_smooth'] - df1['Std_smooth'],
                    df1['Avg_smooth'] + df1['Std_smooth'], color='blue', alpha=0.05, label=f'Std - zen garden')

    # Plot the smoothed 'Avg' line for the second DataFrame
    ax.plot(df2['time'], df2['Avg_smooth'], label=f'Avg - neutral city (smoothed)', linewidth=1.5, color='red')

    # Fill the area between smoothed 'Avg' and smoothed 'Std' for the second DataFrame with a shaded region
    ax.fill_between(df2['time'], df2['Avg_smooth'] - df2['Std_smooth'],
                    df2['Avg_smooth'] + df2['Std_smooth'], color='red', alpha=0.05, label=f'Std - neutral city')

    # Highlight the epoch on the 'Avg - Zen Garden' plot with a shaded region
    ymin_norm = (ymin_value - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ymax_norm = (ymax_value - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])

    ax.axvspan(xmin=highlight_epoch_start, xmax=highlight_epoch_end, ymin=ymin_norm, ymax=ymax_norm,
               color='green', alpha=0.2, label='Breathing epoch during zen garden')

    ax.set_xlabel("Time (seconds)", size=16)
    ax.set_ylabel(f"{column_of_interest} Value", size=16)

    # Set y-axis limits to a larger range if needed
    # ax.set_ylim(bottom=-50, top=150)

    # Place the legend in a separate box on the right
    ax.legend(loc='upper right')

    ax.set_title(f"{column_of_interest} average comparison between zen garden and neutral city videos", size=20)

    plt.show()


import pandas as pd
import matplotlib.pyplot as plt

def plot_comparison_between_part(df1, df2, column_of_interest, ymin_value=18, ymax_value=125, highlight_epoch_start=128, highlight_epoch_end=225):
    """plot to compare the average of a column of interest between zen garden and neutral city videos

    Parameters
    ----------
    df1 : dataframe
        zen garden dataframe
    df2 : dataframe
        neutral city dataframe
    column_of_interest : str
        name of the column of interest (e.g. this time 'LF', 'HF', 'LF_HF_ratio'...)
    ymin_value : int, optional
        min range of breathing epoch, by default 18
    ymax_value : int, optional
        max range of breathing epoch, by default 125
    highlight_epoch_start : int, optional
        start of breathin epoch, by default 128
    highlight_epoch_end : int, optional
        end of breathing epoch, by default 225
    """
    # Calculate rolling mean for 'Std' in the first DataFrame
    df1['Std_smooth'] = df1['Std'].rolling(window=5, min_periods=1).mean()

    # Calculate rolling mean for 'Std' in the second DataFrame
    df2['Std_smooth'] = df2['Std'].rolling(window=5, min_periods=1).mean()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot the 'Avg' line for the first DataFrame
    ax.plot(df1['time'], df1['Avg'], label=f'Avg - relaxed participants', linewidth=2, color='blue')

    # Fill the area between 'Avg' and smoothed 'Std' for the first DataFrame with a shaded region
    ax.fill_between(df1['time'], df1['Avg'] - df1['Std_smooth'],
                    df1['Avg'] + df1['Std_smooth'], color='blue', alpha=0.05, label=f'Std - relaxed participants')

    # Plot the 'Avg' line for the second DataFrame
    ax.plot(df2['time'], df2['Avg'], label=f'Avg - unrelaxed participants', linewidth=2, color='red')

    # Fill the area between 'Avg' and smoothed 'Std' for the second DataFrame with a shaded region
    ax.fill_between(df2['time'], df2['Avg'] - df2['Std_smooth'],
                    df2['Avg'] + df2['Std_smooth'], color='red', alpha=0.05, label=f'Std - unrelaxed participants')

    # Highlight the epoch on the 'Avg - Zen Garden' plot with a shaded region
    ymin_norm = (ymin_value - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    ymax_norm = (ymax_value - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])

    ax.axvspan(xmin=highlight_epoch_start, xmax=highlight_epoch_end, ymin=ymin_norm, ymax=ymax_norm,
               color='green', alpha=0.2, label='Breathing epoch during zen garden')

    ax.set_xlabel("Time (seconds)", size=16)
    ax.set_ylabel(f"{column_of_interest} Value", size=16)

    # Set y-axis limits to a larger range if needed
    # ax.set_ylim(bottom=-50, top=150)

    # Place the legend in a separate box on the right
    ax.legend(loc='upper right')

    ax.set_title(f"{column_of_interest} average comparison between relaxed and unrelaxed participants", size=20)

    plt.show()



