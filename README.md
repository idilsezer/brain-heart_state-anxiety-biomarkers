# **Enhanced Brain-Heart Connectivity as a Precursor to Reduced State Anxiety After Therapeutic Virtual Reality Immersion**
 
## **1. Introduction**
This repository contains pipelines of EEG and ECG, as well as EEG-ECG coupling analyses developed to investigate the therapeutic effects of a virtual reality environment on state anxiety levels. The pipeline handles preprocessing steps and visualisation of data.

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
│   └── /xx/                  # xx
│   └── /xx/               # xx
│
├── /src/                    # Functions and codes
│   ├── bh_biomarkers_figures.ipynb
│   ├── data_processing.py 
│   ├── plot_utils.py
│   └── video_analyses.py
│
├── requirements.txt                # Python dependencies file
```

**Important codes for the analysis**
1. **Psychometric data**

    **`EEG_behavioural_analysis.ipynb`**: Analyses time taken to complete the task and success rate for each Field of View (20°/45°).

2. **EEG data**
   
xx

3. **ECG data**

xx



## **5. Authors, Credits, and Acknowledgments**
- Authors and collaborators: 
- Institutions and companies:
    - [Healthy Mind](https://healthymind.fr/)
    - [Frontlab, Institut du Cerveau](https://institutducerveau.org/equipes-recherche-linstitut-cerveau/frontlab-cortex-prefrontal-au-centre-fonctions-cognitives-superieures-sante-maladie)
- Credits: 
