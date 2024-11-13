import sys
sys.path.append("../../src")

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.collections import PolyCollection

from scipy.stats import wilcoxon, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
# import statsmodels.formula.api as smf
# import statsmodels.api as sm


import video_analyses
import mne

# import warnings
# warnings.filterwarnings("ignore")

# FOR NATURE COMMUNICATIONS GUIDELINES
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300





# TEMPORAL PLOTS (ECG, EEG)

def temporal_plot_comparison(df1, df2, region, band, participants, type, leg_pos, highlight_epoch_start=128, highlight_epoch_end=225, smoothness=2, n=27):
    """plot to compare the average of a column of interest between zen garden and control cities videos

    Parameters
    ----------
    df1 : dataframe
        zen garden dataframe
    df2 : dataframe
        control cities dataframe
    region : str
        name of the region to look at for EEG, or the metric for ECG (LF, HF, fratio)
    band : str
        name of the frequency band to look at for EEG
    participants : str
        either all, unresponsive or responsive (for the title)
    type : str
        EEG or ECG (information for "if" parts)
    leg_pos : str
        Position of the legend
    highlight_epoch_start : int, optional
        start of breathin epoch, by default 128
    highlight_epoch_end : int, optional
        end of breathing epoch, by default 225
    smoothness : int, optional
        window size for smoothing, by default 2
    """

    palette = sns.color_palette("Set2")
    plt.style.use('seaborn-white')


    gris_bleu = tuple(max(0, val - 0.1) for val in palette[2])
    grey = (0.7, 0.7, 0.7)

    if type == 'ECG': # for LF, HF and fratio
        df1['Std_smooth'] = df1['Std'].rolling(window=smoothness, min_periods=1).mean() / np.sqrt(n)
        df2['Std_smooth'] = df2['Std'].rolling(window=smoothness, min_periods=1).mean() / np.sqrt(n)
        df1['Avg_smooth'] = df1['Avg'].rolling(window=smoothness, min_periods=1).mean()
        df2['Avg_smooth'] = df2['Avg'].rolling(window=smoothness, min_periods=1).mean()
    else: 
        # to directly extract the right frequency band and region from the dataframes (EEG)
        df1['Std_smooth'] = df1[region+"_"+band+"_std"].rolling(window=smoothness, min_periods=1).mean() / np.sqrt(n)
        df2['Std_smooth'] = df2[region+"_"+band+"_std"].rolling(window=smoothness, min_periods=1).mean() / np.sqrt(n)
        df1['Avg_smooth'] = df1[region+"_"+band+"_mean"].rolling(window=smoothness, min_periods=1).mean()
        df2['Avg_smooth'] = df2[region+"_"+band+"_mean"].rolling(window=smoothness, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(22, 12))

    # Find the actual time values corresponding to highlight_epoch_start and highlight_epoch_end
    start_time = df1['time'].iloc[highlight_epoch_start]
    end_time = df1['time'].iloc[highlight_epoch_end]

    # Adjust plot slicing using actual time values
    ax.plot(df1['time'][df1['time'] <= end_time], df1['Avg_smooth'][df1['time'] <= end_time], linewidth=2.5, color=palette[7])

    # Similarly adjust the axvspan
    ax.axvspan(xmin=start_time, xmax=end_time, color=palette[2], alpha=0.5, zorder=-1, label='Breathing epoch during zen garden')

    ax.plot(df1['time'][df1['time'] <= end_time], df1['Avg_smooth'][df1['time'] <= end_time], 
        linewidth=2.5, color=grey)

    ax.fill_between(df1['time'][df1['time'] <= end_time], 
                    df1['Avg_smooth'][df1['time'] <= end_time] - df1['Std_smooth'][df1['time'] <= end_time],
                    df1['Avg_smooth'][df1['time'] <= end_time] + df1['Std_smooth'][df1['time'] <= end_time], 
                    color=palette[7], alpha=0.25)


    # NORMAL ZEN GARDEN (after end of breathing)
    ax.plot(df1['time'][highlight_epoch_end:], df1['Avg_smooth'][highlight_epoch_end:], 
            label='Avg - zen garden', linewidth=2.5, color=palette[0])
    ax.fill_between(df1['time'][highlight_epoch_end:], 
                    df1['Avg_smooth'][highlight_epoch_end:] - df1['Std_smooth'][highlight_epoch_end:],
                    df1['Avg_smooth'][highlight_epoch_end:] + df1['Std_smooth'][highlight_epoch_end:], 
                    color=palette[0], alpha=0.25, label='SEM - zen garden')

    # CONTROL CONDITION PLOT
    ax.plot(df2['time'], df2['Avg_smooth'], label=f'Avg - control cities', linewidth=2.5, color=palette[1])
    ax.fill_between(df2['time'], df2['Avg_smooth'] - df2['Std_smooth'],
                     df2['Avg_smooth'] + df2['Std_smooth'], color=palette[1], alpha=0.25, label=f'SEM - control cities')

    ax.fill_between(df1['time'][(df1['time'] >= start_time) & (df1['time'] <= end_time)],
                df1['Avg_smooth'][(df1['time'] >= start_time) & (df1['time'] <= end_time)],
                df2['Avg_smooth'][(df1['time'] >= start_time) & (df1['time'] <= end_time)],
                color=gris_bleu, zorder=2, label='Breathing epoch during zen garden')


    # LABELS, TITLE AND LEGENDS    
    band_label = f"$\\{band}$ " if band else ""
    ylabel = f"{region} {band} (n.u.)" if type == 'ECG' else f"Relative power spectral density for {region} $\\{band}$ (n.u.)"
    ax.set_ylabel(ylabel, size=28, weight='bold', labelpad=15)
    ax.set_xlabel("Time (s)", size=28, weight='bold', labelpad=15)

    # y-axis range
    if type == 'EEG':
        title_start = f"$\\{band}$"
        if band == 'beta':
            ax.set_ylim([0.26, 0.32]) # EEG beta
        elif band == 'alpha':
            ax.set_ylim([0.20, 0.26]) # PAUL ???
    elif type == 'ECG':
        if region == 'HF':
            ax.set_ylim([0, 0.4]) # ECG HF
        elif region == "LF" or region == 'LF/HF':
            ax.set_ylim([0, 0.8]) # ECG LF/HF and LF

        if region == 'LF/HF':
            title_start = "LF/HF ratio"
        else:
            title_start = region

    # ax.set_title(f"Comparison of {region} {band_label}average between zen garden and neutral city videos ({participants} participants)", size=23, weight='bold', pad=30)
    ax.set_title(f"midline {title_start} - {participants} participants", size=35, weight='bold', pad=30)


    ax.spines['top'].set_visible(False) # remove the Figure frame and only keep the x/y axes
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1) 
    ax.spines['left'].set_linewidth(1)  
    ax.tick_params(axis='both', which='major', labelsize=28) 
    ax.grid(alpha=0.4, zorder=-2)

    # reordering legends
    handles, labels = ax.get_legend_handles_labels()
    new_order = [1, 2, 3, 4, 0]
    handles = [handles[i] for i in new_order]
    labels = [labels[i] for i in new_order]
    ax.legend(handles, labels, loc=leg_pos, ncol=3, frameon=True, fontsize=26)

    #plt.savefig(address_output+f"/Figures/{band}_{region}_{participants}.png", bbox_inches='tight', dpi=250) #,'Figures/'+title+'.png'

    plt.show()









################### VIOLIN EEG DATA (REGIONS) ###################

def violin_plots_EEG_regions(df, group, subtitle, band='alpha'):
    # PLOT ALL THE REGIONS FOR THE DESIRED BAND

    if group is None:
        df_sep = df
    else:
        df_sep = df.loc[df['group'] == group]

    metrics_list = df_sep['variable'].unique()  # Use df_sep instead of df to get unique metrics in the filtered dataframe

    num_metrics = len(metrics_list)
    num_cols = 3  # Set the number of columns for subplots
    num_rows = (num_metrics + num_cols - 1) // num_cols  # Calculate the number of rows required

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 10))

    # Turn off the extra subplots if they exceed the number of metrics
    if num_rows * num_cols > num_metrics:
        for i in range(num_metrics, num_rows * num_cols):
            axes.flatten()[i].axis('off')

    axes = axes.flatten()[:num_metrics]  # Flatten and take only the required number of subplots
    
    subplots_letters = ['a','b','c','d','e','f','g']

    for i, metric_name in enumerate(metrics_list):
        violin_plots_EEG_one_region(metric_name, df_sep, subplots_letters[i], ax=axes[i], hue_labels=['Zen Garden', 'Control cities'], band=band)

    # Adjust layout
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
    plt.subplots_adjust(hspace=0.3)  # Increase spacing between rows of subplots
    plt.suptitle('Comparison for '+subtitle+' participants', y=1.01, weight='bold',size=19)
    #plt.savefig(address_output+'/Figures/violinplots_'+subtitle+'.png', bbox_inches='tight', dpi=250)
    plt.show()


def violin_plots_EEG_one_region(metric_name, melted_data_resp, title_letter, ax, hue_labels, band='alpha'):
    # PLOT ONE REGION FOR THE DESIRED BAND (called by the function above)

    sns.set(style="white")
    # Filter the melted_data for the current metric
    metric_data = melted_data_resp.loc[melted_data_resp['variable'] == metric_name]

    # Convert 'condition' to string type
    metric_data['condition'] = metric_data['condition'].astype(str)
    palette = sns.color_palette("Set2")
    palette_inv = palette[:2][::-1] # to inverse colours

    ax = sns.violinplot(data=metric_data, x='condition', y='value',order=['1','0'], dodge=True, saturation=1,
                        palette=palette_inv, linewidth=0.5, ax=ax)
    
    # Customize violin plots
    colors = []
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            colors.append(collection.get_facecolor())
            collection.set_edgecolor(colors[-1])
            collection.set_facecolor(to_rgba(colors[-1], alpha=0.4))  # change transparency of background

    # Adjust the lines
    if len(ax.lines) == 2 * len(colors):  # suppose inner=='quartile'
        for lin1, lin2, color in zip(ax.lines[::2], ax.lines[1::2], colors):
            lin1.set_color(color)
            lin2.set_color(color)
            lin1.set_linewidth(2)  # boxplot's stick width
            lin2.set_linewidth(8)  # boxplot width

    # Connect dots and draw lines between conditions 0 and 1 for each ID within each group
    for i in range(len(metric_data)):
        id_val = metric_data.iloc[i]['ID']
        cond_0_val = metric_data.loc[(metric_data['ID'] == id_val) & (metric_data['condition'] == '0')]['value'].values
        cond_1_val = metric_data.loc[(metric_data['ID'] == id_val) & (metric_data['condition'] == '1')]['value'].values

        if len(cond_0_val) > 0 and len(cond_1_val) > 0:  # Ensure both conditions have values
            cond_0_val = cond_0_val[0]  # Take the first value
            cond_1_val = cond_1_val[0]  # Take the first value

            x_pos = [0.1, 0.9]  # Adjust x-coordinates for dot alignment
            ax.plot(x_pos, [cond_1_val, cond_0_val], marker='o', color=sns.color_palette("husl")[4], linestyle='-', linewidth=0.3, alpha=1, markersize=2)

    # Customize plot labels and title
    ax.set_title(metric_name, weight='bold', size=18)
    ax.set_xlabel("Condition")
    ax.set_ylabel("(n.u.)")
    ax.set_xticklabels(['Control cities','Zen Garden'])
    
    
    # Set different y-axis limits based on the band
    # if band == 'alpha':
    #     ax.set_ylim([0.125, 0.315])
    # elif band == 'beta':
    #     ax.set_ylim([0.25, 0.35])
    if band == 'alpha':
        ax.set_ylim([-6,6]) #before: -6,5
    elif band == 'beta':
        ax.set_ylim([-5, 5]) #before: -4,5

    ax.spines['top'].set_visible(False)  # remove the Figure frame and only keep the axes
    ax.spines['right'].set_visible(False)

    ax.grid(alpha=0.4)

    # TEST FOR SUBPLOTS LETTERS a, b, c ...
    ax.text(-0.1, 1.05, title_letter, transform=ax.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')

    # Calculate Wilcoxon signed-rank test p-values
    p_values = []
    for region in metric_data['variable'].unique():
        region_data = metric_data[metric_data['variable'] == region]
        _, p_val = wilcoxon(region_data[region_data['condition'] == '0']['value'],
                             region_data[region_data['condition'] == '1']['value'])
        p_values.append(p_val)

    # Apply False Discovery Rate (FDR) correction to p-values
    corrected_p_values = multipletests(p_values, method='fdr_bh')[1]

    # Determine significance levels based on corrected p-values
    significance_levels = ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.' for p in corrected_p_values]

    # Annotate violin plots with significance levels ## COMMENTED
    # for i, label in enumerate(significance_levels):
    #     if band == 'alpha':
    #         #ax.text(i, 0.3, label, ha='center', va='center', fontweight='bold', color='black')
    #         ax.text(i, 4, label, ha='center', va='center', fontweight='bold', color='black')
    #     if band == 'beta':
    #         #ax.text(i, 0.345, label, ha='center', va='center', fontweight='bold', color='black')
    #         ax.text(i, 4, label, ha='center', va='center', fontweight='bold', color='black')










################### VIOLIN STAIY COMPARISON ###################

def violin_plots_staiy(df):
    sns.set(style="white")
    palette = sns.color_palette("Set2")
    darker_color = tuple(min(1, c - 0.4) for c in palette[0])  # Adjust the factor (0.3) for lighter or darker shade
    lighter_color = tuple(min(1, c + 0.1) for c in palette[0])  # Adjust the factor (0.3) for lighter or darker shade
    custom_palette = [lighter_color, darker_color]

    fig = plt.figure(figsize=(11, 8))

    ax = sns.violinplot(data=df, x='group', y='score', hue='order', dodge=True, saturation=1,palette=custom_palette, linewidth=0.5)

    # Customize violin plots
    colors = []
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            colors.append(collection.get_facecolor())
            collection.set_edgecolor(colors[-1])
            collection.set_facecolor(to_rgba(colors[-1], alpha=0.4))  # change transparency of background

    # Adjust the lines
    if len(ax.lines) == 2 * len(colors):  # suppose inner=='quartile'
        for lin1, lin2, color in zip(ax.lines[::2], ax.lines[1::2], colors):
            lin1.set_color(color)
            lin2.set_color(color)
            lin1.set_linewidth(2)  # boxplot's stick width
            lin2.set_linewidth(8)  # boxplot width

    # Connect dots and draw lines between conditions 0 and 1 for each ID within each group
    for group_num in df['group'].unique():
        group_data = df.loc[df['group'] == group_num]
        for i in range(len(group_data)):
            id_val = group_data.iloc[i]['Participant']
            cond_0_val = group_data.loc[(group_data['Participant'] == id_val) & (group_data['order'] == '0')]['score'].values
            cond_1_val = group_data.loc[(group_data['Participant'] == id_val) & (group_data['order'] == '1')]['score'].values

            x_pos = [group_num - 0.15, group_num + 0.15]  # Adjust x-coordinates for dot alignment
            ax.plot(x_pos, [cond_0_val, cond_1_val], marker='o', color = sns.color_palette("husl")[4],
                    linestyle='-', linewidth=0.3, alpha=1, markersize=2)

    # Adjust legend handles and labels
    for i, (h, label) in enumerate(zip(ax.legend_.legendHandles, ['Pre score', 'Post score'])):
        ax.legend_.get_texts()[i].set_text(label)
        ax.legend_.get_texts()[i].set_fontsize(16) # added to increase legend size

    for text in ax.legend_.get_texts():
        text.set_fontsize(16)
    
    ax.legend_.set_frame_on(True)

    # Customize plot labels and title
    ax.set_title("Comparison of STAIY scores pre/post Zen garden, for unresponsive & responsive participants",
                weight='bold', size=18, y=1.05)
    ax.set_xlabel("Group", size=18)
    ax.set_ylabel("STAIY score", size=18)
    ax.set_xticklabels(['Responsive', 'Unresponsive'], size=16)

    ax.spines['top'].set_visible(False)  # remove the Figure frame and only keep the axes
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.4)

    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.show()
    # plt.savefig(address_output+'/Figures/staiy.png', bbox_inches='tight', dpi=250)


def plot_baseline_staiy(df, group=None, subtitle=None):
    if group is not None:
        df = df.loc[df['group'] == group]

    sns.set(style="white")
    palette = sns.color_palette("Set2")
    darker_color = tuple(min(1, c - 0.4) for c in palette[0])  # Adjust the factor (0.3) for lighter or darker shade
    lighter_color = tuple(min(1, c + 0.1) for c in palette[0])  # Adjust the factor (0.3) for lighter or darker shade
    custom_palette = [lighter_color, darker_color]

    plt.figure(figsize=(11, 8))

    ax = sns.violinplot(data=df, y='before', x='group', dodge=True, saturation=1,
                        palette=custom_palette, linewidth=0.5) #inner_kws=dict(box_width=15, whis_width=2)

    colors = []
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            colors.append(collection.get_facecolor())
            collection.set_edgecolor(colors[-1])
            collection.set_facecolor(to_rgba(colors[-1], alpha=0.4)) # change transparency of background

    # Adjust the lines
    if len(ax.lines) == 2 * len(colors):  # suppose inner=='quartile'
        for lin1, lin2, color in zip(ax.lines[::2], ax.lines[1::2], colors):
            lin1.set_color(color)
            lin2.set_color(color)
            lin1.set_linewidth(2) #boxplot's stick width
            lin2.set_linewidth(8) #boxplot width

    # Connect dots and draw lines between conditions 0 and 1 for each ID within each group
    # for i in range(len(df)):
    cond_0_val = df.loc[(df['group'] == '0')]['before'].values
    cond_1_val = df.loc[(df['group'] == '1')]['before'].values

    ax.plot([0.1,0.9], [cond_0_val, cond_1_val], marker='o', color=sns.color_palette("husl")[4], linestyle='-', linewidth=0.3, alpha=1, markersize=2) 

    if subtitle is not None:
        ax.set_title("STAI-Y1 scores of " + subtitle + " participants", weight='bold')
    else:
        ax.set_title("baseline STAIY scores of groups", weight='bold')
    ax.set_xlabel("STAI-Y1 baseline for each group")
    ax.set_ylabel("STAI-Y1 score")
    ax.set_ylim([5,90])

    ax.spines['top'].set_visible(False) # remove the Figure frame and only keep the axes
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()






################### VIOLIN METRICS AND STAIY ###################

def violin_plots_metrics_subplots(metric_name, melted_data_resp, title_letter, ax, hue_labels):
    sns.set(style="white")
    palette = sns.color_palette("Set2")
    palette_inv = palette[:2][::-1] # to inverse colours

    metric_data = melted_data_resp.loc[melted_data_resp['variable'] == metric_name] # current metric
    metric_data['condition'] = metric_data['condition'].astype(str)

    ax = sns.violinplot(data=metric_data, x='condition', y='value', order=['1','0'], dodge=True, saturation=1,
                        palette=palette_inv, linewidth=0.5, ax=ax) #inner_kws=dict(box_width=15, whis_width=2) #changed Set2 to custom_palette

    colors = []
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            colors.append(collection.get_facecolor())
            collection.set_edgecolor(colors[-1])
            collection.set_facecolor(to_rgba(colors[-1], alpha=0.4)) # change transparency of background

    if len(ax.lines) == 2 * len(colors):  # suppose inner=='quartile'
        for lin1, lin2, color in zip(ax.lines[::2], ax.lines[1::2], colors):
            lin1.set_color(color)
            lin2.set_color(color)
            lin1.set_linewidth(2) #boxplot's stick width
            lin2.set_linewidth(8) #boxplot width

    # Connect dots and draw lines between conditions 0 and 1 for each ID within each group
        for i in range(len(metric_data)):
            id_val = metric_data.iloc[i]['ID']
            cond_0_val = metric_data.loc[(metric_data['ID'] == id_val) & (metric_data['condition'] == '0')]['value'].values
            cond_1_val = metric_data.loc[(metric_data['ID'] == id_val) & (metric_data['condition'] == '1')]['value'].values

            x_pos = [0.1, 0.9] 
            ax.plot(x_pos, [cond_1_val, cond_0_val], marker='o', color=sns.color_palette("husl")[4], linestyle='-', linewidth=0.3, alpha=1, markersize=2) 
            
    ax.set_title(metric_name, weight='bold', size=20)
    ax.set_xlabel("Condition", size=18)
    ax.set_ylabel("(n.u.)", size=18)
    ax.set_xticklabels(['Control Cities','Zen Garden'], size=18)

    ax.spines['top'].set_visible(False) # remove the Figure frame and only keep the axes
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.4)
    ax.text(-0.1, 1.05, title_letter, transform=ax.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')




def violin_plots_staiy_subplot(df, ax, title_letter, group=None, subtitle=None, ):
    if group is not None:
        df = df.loc[df['group'] == group]

    sns.set(style="white")
    palette = sns.color_palette("Set2")
    darker_color = tuple(min(1, c - 0.4) for c in palette[0])  # Adjust the factor for lighter or darker shade
    lighter_color = tuple(min(1, c + 0.1) for c in palette[0]) 
    custom_palette = [lighter_color, darker_color]
    
    sns.violinplot(data=df, y='score', x='order', dodge=True, saturation=1, palette=custom_palette, linewidth=0.5, ax=ax)  
    
    colors = []
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            colors.append(collection.get_facecolor())
            collection.set_edgecolor(colors[-1])
            collection.set_facecolor(to_rgba(colors[-1], alpha=0.4))  # change transparency of background

    if len(ax.lines) == 2 * len(colors):  # suppose inner=='quartile'
        for lin1, lin2, color in zip(ax.lines[::2], ax.lines[1::2], colors):
            lin1.set_color(color)
            lin2.set_color(color)
            lin1.set_linewidth(2)  # boxplot's stick width
            lin2.set_linewidth(8)  # boxplot width

    # Connect dots and draw lines between conditions 0 and 1 for each ID within each group
    cond_0_val = df.loc[(df['order'] == '0')]['score'].values
    cond_1_val = df.loc[(df['order'] == '1')]['score'].values
    print(cond_0_val, cond_1_val)

    ax.plot([0.1,0.9], [cond_0_val, cond_1_val], marker='o', color=sns.color_palette("husl")[4], linestyle='-', linewidth=0.3, alpha=1, markersize=2) 

    ax.set_title(f"STAIY {'scores of all participants' if subtitle is None else ''}", weight='bold', size=17)
    ax.set_xlabel("Pre / post Zen Garden score", size=18)
    ax.set_ylabel("STAIY score", size=15)
    ax.set_ylim([5,90])

    ax.spines['top'].set_visible(False)  # remove the Figure frame and only keep the axes
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.4)
    ax.set_xticklabels(['Pre','Post'], size=18)
    ax.text(-0.1, 1.05, title_letter, transform=ax.transAxes, fontsize=22, fontweight='bold', va='top', ha='right')



def violin_plots_metrics_and_staiy(df, df_staiy, group, subtitle, old = 1):
    custom_titles = ["STAI-Y1 scores",  "midline "+f"$\\beta$", "midline " +f"$\\alpha$", "LF", "HF", "LF/HF"]
    metrics_list = ['midline_beta', 'midline_alpha', 'LF_avg', 'HF_avg', 'Fratio_avg']

    if group is None :
        df_sep = df
    else:
        df_sep = df.loc[df['group'] == group]

    if old:
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
        axes[2, 0].axis('off')
        axes[2, 2].axis('off')
    else:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 12))

    axes = axes.flatten()

    # Select elements using the indices to keep
    indices_to_keep = [idx for idx in range(len(axes)) if idx != 6 and idx != 8]
    axes = axes[indices_to_keep]
    subplots_letters = ['a','b','c','d','e','f','g'] # 

    # PLOT STAIY IN SUBPLOT a.
    violin_plots_staiy_subplot(df_staiy, axes[0], subplots_letters[0], group=group, subtitle=subtitle)
    axes[0].set_title(custom_titles[0], fontsize=16, weight='bold')

    # PLOT METRICS IN OTHER SUBPLOTS
    for i, metric_name in enumerate(metrics_list):
        if i < len(axes):
            ax = axes[i+1]
            violin_plots_metrics_subplots(metric_name, df_sep, subplots_letters[i+1], ax=ax, hue_labels=['Zen Garden', 'Control video'])
            ax.set_title(custom_titles[i+1], fontsize=16, weight='bold')

            if metric_name in ['midline_alpha', 'midline_beta']:
                ax.set_ylabel('Relative spectral power (n.u.)')
            else:
                ax.set_ylabel('(n.u.)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
    plt.subplots_adjust(hspace=0.3)  # Increase spacing between rows of subplots
    plt.suptitle('Comparison for '+subtitle+' participants', y = 1.01, weight='bold',size=19)
    # plt.savefig(address_output+'/Figures/violinplots_'+subtitle+'.png', bbox_inches='tight', dpi=250)
    plt.show()


# def violin_plots_metrics_and_staiy_subgroups(df, df_staiy, group, subtitle, old = 1):
#     custom_titles = ["STAI-Y1 scores",  "midline "+f"$\\beta$", "midline " +f"$\\alpha$", "LF", "HF", "LF/HF"]
#     metrics_list = ['midline_beta', 'midline_alpha', 'LF_avg', 'HF_avg', 'Fratio_avg']

#     if group is None :
#         df_sep = df
#     else:
#         df_sep = df.loc[df['group'] == group]

#     if old:
#         fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
#         axes[2, 0].axis('off')
#         axes[2, 2].axis('off')
#     else:
#         fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 15))

#     axes = axes.flatten()

#     # Select elements using the indices to keep
#     indices_to_keep = [idx for idx in range(len(axes)) if idx != 6 and idx != 8]
#     axes = axes[indices_to_keep]
#     subplots_letters = [' ',' ',' ',' ',' ',' ',' '] #['a','b','c','d','e','f','g'] # 

#     # PLOT STAIY IN SUBPLOT a.
#     violin_plots_staiy_subplot(df_staiy, axes[0], subplots_letters[0], group=group, subtitle=subtitle)
#     axes[0].set_title(custom_titles[0], fontsize=16, weight='bold')

#     # PLOT METRICS IN OTHER SUBPLOTS
#     for i, metric_name in enumerate(metrics_list):
#         if i < len(axes):
#             ax = axes[i+1]
#             violin_plots_metrics_subplots(metric_name, df_sep, subplots_letters[i+1], ax=ax, hue_labels=['Zen Garden', 'Control video'])
#             ax.set_title(custom_titles[i+1], fontsize=16, weight='bold')

#             if metric_name in ['midline_alpha', 'midline_beta']:
#                 ax.set_ylabel('Relative spectral power (n.u.)')
#             else:
#                 ax.set_ylabel('(n.u.)')

#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
#     plt.subplots_adjust(hspace=0.3)  # Increase spacing between rows of subplots
#     plt.suptitle('Comparison for '+subtitle+' participants', y = 1.01, weight='bold',size=19)
#     # plt.savefig(address_output+'/Figures/violinplots_'+subtitle+'.png', bbox_inches='tight', dpi=250)
#     plt.show()

def violin_plots_metrics_and_staiy_subgroups(df, df_staiy, group, subtitle, old=1):
    custom_titles = ["STAI-Y1 scores",  "midline " + f"$\\beta$", "midline " + f"$\\alpha$", "LF", "HF", "LF/HF"]
    metrics_list = ['midline_beta', 'midline_alpha', 'LF_avg', 'HF_avg', 'Fratio_avg']

    if group is None:
        df_sep = df
    else:
        df_sep = df.loc[df['group'] == group]

    if old:
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
        axes[2, 0].axis('off')
        axes[2, 2].axis('off')
    else:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 15))

    axes = axes.flatten()

    # Select elements using the indices to keep
    indices_to_keep = [idx for idx in range(len(axes)) if idx != 6 and idx != 8]
    axes = axes[indices_to_keep]
    subplots_letters = [' ', ' ', ' ', ' ', ' ', ' ', ' '] 

    # PLOT STAIY IN SUBPLOT a.
    violin_plots_staiy_subplot(df_staiy, axes[0], subplots_letters[0], group=group, subtitle=subtitle)
    axes[0].set_title(custom_titles[0], fontsize=20, weight='bold')

    # Apply x-axis label adjustments to add padding in the STAIY subplot
    axes[0].set_xticks([0, 1])  # Assuming Zen Garden = 0 and Control video = 1
    axes[0].set_xticklabels(['  Pre score  ', '  Post score  '], rotation=0, ha='center')  # Add whitespace padding
    axes[0].tick_params(axis='x', pad=10)  # Adjust pad for more spacing if needed
    axes[0].tick_params(axis='y', labelsize=18)
    axes[0].set_ylabel("STAI-Y1 score", fontsize=20)


    # PLOT METRICS IN OTHER SUBPLOTS
    for i, metric_name in enumerate(metrics_list):
        if i < len(axes):
            ax = axes[i+1]
            violin_plots_metrics_subplots(metric_name, df_sep, subplots_letters[i+1], ax=ax, hue_labels=['  Control cities  ', '  Zen Garden  '])
            ax.set_title(custom_titles[i+1], fontsize=22, weight='bold')

            if metric_name in ['midline_alpha', 'midline_beta']:
                ax.set_ylabel('Relative spectral power (n.u.)', fontsize=20)
            else:
                ax.set_ylabel('(n.u.)', fontsize=20)

            # Add padding between the x-axis labels "Condition" and the individual labels
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['  Control cities  ', '  Zen Garden  '], rotation=0, ha='center', fontsize=18)
            ax.tick_params(axis='x', pad=10)
            ax.set_xlabel("Condition", fontsize=20)
            ax.tick_params(axis='y', labelsize=16)


    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
    plt.subplots_adjust(hspace=0.3)  # Increase spacing between rows of subplots
    plt.suptitle('Comparison for ' + subtitle + ' participants', y=1.01, weight='bold', size=19)
    plt.show()





################### TOPOPLOTS ###################

def plot_individual_topomap(regions, frontal_value, temporal_value, central_value, parietal_value, occipital_value, midline_value, vlim):
    """
    Creates and displays an EEG topomap with specified region values.
    
    Arguments:
    - regions: dict containing lists of channel names for each region.
    - frontal_value, temporal_value, central_value, parietal_value, occipital_value, midline_value: values to assign to each region.
    - vlim: tuple defining the color scale limits.
    - cmap: colormap used for the topomap.
    - figsize: figure size.
    """

    signal_info = mne.create_info(
        regions['frontal'] + regions['temporal'] + regions['central'] + regions['parietal'] + regions['occipital'] + regions['midline'], 
        250, ch_types='eeg'
    )
    
    ten_twenty_montage = mne.channels.make_standard_montage('easycap-M1')
    signal_info.set_montage(ten_twenty_montage)
    
    values = (
        [frontal_value] * len(regions['frontal']) +
        [temporal_value] * len(regions['temporal']) +
        [central_value] * len(regions['central']) +
        [parietal_value] * len(regions['parietal']) +
        [occipital_value] * len(regions['occipital']) +
        [midline_value] * len(regions['midline'])
    )
    
    fig, ax = plt.subplots(figsize=(10, 6)) 
    
    im, _ = mne.viz.plot_topomap(values, signal_info, axes=ax, show=False, vlim=vlim, cmap='coolwarm', contours=0)
    
    plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04, format='%.2f')
    plt.tight_layout()
    plt.show()