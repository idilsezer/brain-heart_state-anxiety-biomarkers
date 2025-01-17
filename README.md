# **Enhanced Brain-Heart Connectivity as a Precursor to Reduced State Anxiety After Therapeutic Virtual Reality Immersion**
 Idil Sezer, Paul Moreau, Mohamad El Sayed Hussein Jomaa, Valérie Godefroy, Bénédicte Batrancourt, Richard Lévy, Anton Filipchuk          
 bioRxiv 2024.11.28.625818; doi: https://doi.org/10.1101/2024.11.28.625818
 **For reuse, please cite: [Preprint](https://doi.org/10.1101/2024.11.28.625818)**

 ![Summary of findings](/Figure7.png)

## **1. Introduction**
This repository contains pipelines of EEG and ECG processing, as well as EEG-ECG coupling analyses developed to investigate the therapeutic effects of a virtual reality environment on state anxiety levels. The pipeline handles processing steps and visualisation of data.

The project is conducted within the Healthy Mind company and the Frontlab at the Paris Brain Institute Institut (Sorbonne University,CNRS/INSERM, Paris, FRANCE).

## **2. Installation**

Follow these steps to set up the project environment:

### Clone the Repository
To start, clone the repository to your local machine:

```console
git clone https://github.com/idilsezer/brain-heart_state-anxiety-biomarkers.git && cd brain-heart_state-anxiety-biomarkers
```

### Create a Virtual Environment
To create a virtual environment, run the following command:

```console
python -m venv bh-sa-env
```

### Activate the Virtual Environment
Activate the virtual environment using one of the following commands based on your operating system:

**Windows:**
```console
.\bh-sa-env\Scripts\activate
```
**Linux / Mac:**
```console
source bh-sa-env/bin/activate
```

### Install Dependencies
Next, install the necessary dependencies by running:

```console
pip install -r requirements.txt
```

### Additional Requirements



## **3. Project Structure**

**Summary of the folders structure**
```
├── /data/
│   ├── /alpha_averages_ctl_dict_ALL.pickle          # dictionary of alpha values in all participants in the control condition
│   ├── /alpha_averages_dict_ALL.pickle              # dictionary of alpha values in all participants in the zen garden condition
│   ├── /beta_averages_ctl_dict_ALL.pickle           # dictionary of beta values in all participants in the control condition
│   ├── /beta_averages_dict_ALL.pickle               # dictionary of beta values in all participants in the zen garden condition
│   ├── /biomarkers_GLMM.xlsx                        # metrics during the zen garden condition and magnitude of response (delta STAI-Y1)
│   ├── /brain_heart_coefficients.csv                # brain-heart coupling coefficients
│   ├── /coeff_all_ctl_stats.csv                     # brain-heart coupling coefficients for statistical analyses, during control condition
│   ├── /coeff_all_zg_seg_stats.csv                  # brain-heart coupling coefficients for statistical analyses, during zen garden condition, 'seg'= without breathing epoch
│   ├── /df_metrics_melted.csv                       # midline beta, midline alpha, LF, HF, LF/HF ratio melted values for violin plots
│   ├── /EEG_beta_regions.csv                        # EEG beta of all regions: frontal, central, temporal, parietal, occipital, midline
│   ├── /EEG_high_alpha_regions_stats.csv            # EEG high alpha (10-13 Hz) of all regions: frontal, central, temporal, parietal, occipital, midline, for statistics (outlier removal, R)
│   ├── /EEG_high_alpha_regions.csv                  # EEG high alpha (10-13 Hz) of all regions: frontal, central, temporal, parietal, occipital, midline, for visualisations (without outlier)
│   ├── /Fratio_normalized_ctl.csv                   # within-subject normalized LF/HF ratio values, control condition
│   ├── /Fratio_normalized_zg.csv                    # within-subject normalized LF/HF ratio values, zen garden condition (breathing epoch removed)
│   ├── /HF_normalized_ctl.csv                       # within-subject normalized HF values, control condition
│   ├── /HF_normalized_zg.csv                        # within-subject normalized HF values, zen garden condition (breathing epoch removed)
│   ├── /LF_normalized_ctl.csv                       # within-subject normalized LF values, control condition
│   ├── /LF_normalized_zg.csv                        # within-subject normalized LF values, zen garden condition (breathing epoch removed)
│   ├── /STAI-Y1-SCORES.csv                          # state anxiety (STAI-Y1) scores of all participants: baseline, before & after zen garden condition
│   └── videos_all_metrics_for_stats.csv             # midline beta, midline alpha, LF/HF ratio, LF, HF values across conditions used for statistical analyses
│
│
├── /src/                     # Functions and codes
│   ├── bh_biomarkers_figures.ipynb                  # visualisations of Figures of the article (Figures 1-5, main text)
│   ├── bh_biomarkes_supplemental.ipynb              # visualisations of Figures of the article (Supplementary Figures 1-8, supplemental)
│   ├── data_processing.py                           # processing of EEG and ECG data
│   └── plot_utils.py                                # plot generation functions
│
├── /statistics/    
│   ├── brain2heart_statistics.R                     # Brain-heart coupling coefficients - statistical analyses
│   └── state-anxiety-statistics.R                   # All statistical analyses and Figure 6 GLMM generation
│
├── requirements.txt          # Python dependencies file
```

**Important codes for the analysis**
1. **EEG, ECG and brain-heart analyses**

    **`bh_biomarkers_figures.ipynb`**: Processing of data and generation of plots: STAI-Y1 scores, midline EEG (including topographic maps), HF, LF, brain-heart coupling

2. **Supplementary analyses**
   
    **`bh_biomarkes_supplemental.ipynb`**: Processing of data and generation of plots: LF, EEG beta and high alpha regions (violin plots and topographical maps)
   
4. **Statistical analyses and GLMM of response biomarker**

    **`state-anxiety-statistics.R`**: All statistical analyses and generation of GLMM to predict magnitude of response (delta STAI-Y1) from LF/HF ratio

    **`brain2heart_statistics.R`**: Calculation of brain-heart coupling statistics
   
## **5. Authors, Credits, and Acknowledgments**
- Authors and collaborators: 
- Institutions and companies:
    - [Healthy Mind](https://healthymind.fr/)
    - [Frontlab, Institut du Cerveau](https://institutducerveau.org/equipes-recherche-linstitut-cerveau/frontlab-cortex-prefrontal-au-centre-fonctions-cognitives-superieures-sante-maladie)
- Credits:
[Preprint](https://doi.org/10.1101/2024.11.28.625818)
