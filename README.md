# **The role of brain-heart connectivity in state anxiety reduction during virtual reality hypnotic immersion**
 
## **1. Introduction**
This repository contains an EEG analysis pipeline developed for investigating the impact of varying fields of view on the effectiveness of the POLYRETINA retinal implant. The pipeline is based on the BeMoBIL framework and is designed to handle preprocessing, analysis, and visualisation of EEG data. The POLYRETINA project aims to assess how different visual angles influence cognitive load and user performance, with the ultimate goal of enhancing autonomy for individuals with acquired blindness.

The project is conducted within the Aging in Vision and Action Laboratory (AVA Lab) at the Institut de la Vision (Paris, FRANCE), in collaboration with the École Polytechnique Fédérale de Lausanne (EPFL - Lausanne, SWITZERLAND).

## **2. Installation**

Follow these steps to set up the project environment:

### Clone the Repository
To start, clone the repository to your local machine:

```console
git clone https://github.com/paulmoreau40/EEG-POLYRETINA.git && cd EEG-POLYRETINA
```

### Create a Virtual Environment
To create a virtual environment, run the following command:

```console
python -m venv polyretina-env
```

### Activate the Virtual Environment
Activate the virtual environment using one of the following commands based on your operating system:

**Windows:**
```console
.\polyretina-env\Scripts\activate
```
**Linux / Mac:**
```console
source polyretina-env/bin/activate
```

### Install Dependencies
Next, install the necessary dependencies by running:

```console
pip install -r requirements.txt
```

### Additional Requirements

**EEGLAB 2024.0**:  
   - Download EEGLAB version 2024.0 from the official website: https://sccn.ucsd.edu/eeglab/download.php.
   - Unzip the downloaded archive.
   - Place the **EEGLAB** folder in the **toolboxes/** directory of the project:
     ```
     /toolboxes/eeglab2024.0/
     ```



## **3. Project Structure**

**Summary of the folders structure**
```
├── /data/
│   └── /analysis/                  # Data for EEG analysis (raw, preprocessed, processed, analysis)
│   └── /behavioural/               # Data for behavioural analysis (trial time, success rate)
│
├── /src/                    # Functions and codes
│   ├── videos_for_Paul_10.ipynb
│   ├── plot_utils.py
│   ├── data_processing.py
│   └── video_analyses.py
│
├── /figures/ 
│   ├── AnalysisPlots               # Outputs of the analysis (heatmaps, power spectra, topoplots)
│   ├── BehaviouralAnalysisPlots    # Outputs of the behavioural analysis (success rate and trial time)
│   ├── ProcessingPlots             # Figures produced during the processing steps
│       ├── /AutoBadSampsCleaning/
│       ├── /ChannelsImport/
│       └── /PREP_distributions/
│
├── /toolboxes/                     # Contains external toolboxes and dependencies
│   ├── /bemobil_pipeline0.2/       # Functions from the BeMoBIL pipeline
│   ├── /eeglab2024.0/              # EEGLAB toolbox for EEG analysis
│   └── /ParforProgMon/             # Parfor Progress Monitor toolbox
│
├── requirements.txt                # Python dependencies file
```

**Important codes for the analysis**
1. **Behavioural data**

    **`EEG_behavioural_analysis.ipynb`**: Analyses time taken to complete the task and success rate for each Field of View (20°/45°).

2. **EEG data**
   
    Mains:
    - **`main_preprocessing.m`**: Imports XDF files, prepares data and preprocesses it, performs ICA and Dipole Fitting, and labels Independent Components (ICs) automatically and manually.
    - **`main_single_participant_analysis.m`**: Conducts main analysis and statistics with data visualisation.

    Configuration:
    - **`ConfigEEGPOL.m`**: Contains study parameters (threshold filters, channels ELC filenames, types of data etc.). Can be modified according to the needs of the study.
    - **`getMainFoldersNames.m`**:  Manages paths to fit the directory structure (data, figures, etc.). Can be personalised if necessary.

    Data preparation and preprocessing:
    - **`xdf2set.m`**: Converts raw XDF files to EEGLAB .set format, processing EEG, motion capture and eye tracking data, events, and channels.
    - **`export_events_EEGPOL.m`**: Imports the events and creates a table with event names, types, latencies, blocks, and trials.
    - **`events_check_EEGPOL.m`**: Checks the integrity of the EEG data for each event.
    - **`keptComponents_POL.m`**: Lists Independent Components that were manually selected (filled by the user).


## **4. Data Requirements**

There are several types of data for different purposes: EEG data, EEG channels information, and metadata, with details on file naming conventions, formats, and directory structures.

### EEG Data
The EEG data for each participant should follow a specific naming convention and file structure. Multiple blocks can be present for each participant, and all EEG data files must be stored in a predefined directory.

- **File format**: `.xdf`
- **File naming convention**: Files must be named as follows:  
  `sub-<ParticipantID>_block<blockNumber>_<session>.xdf`  
  Example: `sub-P001_block001_1.xdf` (first block, first session for participant P001).
  
- **Directory structure**: EEG data files should be stored in the following path:  
  `\data\analysis\0_raw-data\<ParticipantID>`  
  Example: EEG data for participant P001 should be stored in `\data\analysis\0_raw-data\P001`.

- **Multiple blocks**: If multiple blocks are recorded for a participant, each block should have a unique identifier, for example:  
  `sub-P001_block001_1.xdf`  
  `sub-P001_block002_1.xdf`

### EEG Channels Information
In addition to the EEG data, each participant folder must contain a file that describes the EEG channels and their coordinates.
- **File format**: `.elc`
- **File naming convention**: The channels file has a fixed name and is the same for all participants. It must be named:  
  `CA-213_EOG.elc`

- **Directory structure**: The channels file must be present in the following directory:  
  `\data\analysis\0_raw-data\<ParticipantID>`  
  Example: The channels file should be stored in all participant folders, such as `\data\analysis\0_raw-data\P001`.

- **Configuration**: If a different file name is needed, it can be modified in `configEEGPOL.m` by changing the variable `study_config.channel_locations_filename`.


### Metadata
The metadata provides additional information about each participant and must be supplied in a specific Excel file format.

- **File format**: `.xlsx`
- **File name**: `Polyretina_meta.xlsx`
- **Directory structure**: The metadata file should be located in:  
  `\data\analysis`  
  Note: The metadata file can be renamed in the code (for example, via the function `getMainFoldersNames.m`).

- **Metadata structure**: The Excel file should contain a sheet named **SubjectInfo**, and the columns must follow this structure:

  | id   | badElectrodes                  | missingData | excluded |
  |------|--------------------------------|-------------|----------|
  | P001 | L1,LC1,LD1,LL1,R1,RC1,RD1,RR1  | ""          | No       |
  | P002 | L1,LC1,LD1,LL1,R1,RC1,RD1,RR1  | ""          | No       |
  | P003 | L1,LC1,LD1,LL1,R1,RC1,RD1,RR1  | ""          | No       |
  | P004 | L1,LC1,LD1,LL1,R1,RC1,RD1,RR1  | ""          | Yes      |

- **Columns**:
  - `id`: Participant ID, following the format `P<ParticipantNumber>`.
  - `badElectrodes`: List of electrodes marked as bad for the corresponding participant.
  - `missingData`: If there is missing data, it will be noted here (empty string `""` means no missing data).
  - `excluded`: Indicates whether the participant has been excluded from the analysis (`Yes` or `No`).



## **5. Authors, Credits, and Acknowledgments**
- Authors and collaborators: 
- Institutions and companies:
    - [Healthy Mind](https://healthymind.fr/)
    - [Frontlab, Institut du Cerveau](https://institutducerveau.org/equipes-recherche-linstitut-cerveau/frontlab-cortex-prefrontal-au-centre-fonctions-cognitives-superieures-sante-maladie)
- Credits: 
