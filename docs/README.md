# Cognitive Load Experiment

A repository for conducting experiments on **cognitive load** using EEG/EMG measurements, maze-based tasks, and n-back tasks. This project includes data collection scripts, signal-processing pipelines, and visualization tools.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)  
3. [Installation & Requirements](#installation--requirements)  
4. [Data](#data)  
5. [Usage](#usage)  
6. [Key Scripts](#key-scripts)  
7. [Additional Utilities](#additional-utilities)  
8. [License & Citation](#license--citation)

---

## Project Overview

This repository contains:

- **Maze & N-Back** tasks designed to induce varying levels of cognitive load.  
- **Signal Processing** modules for EEG/EMG data analysis (e.g., ICA, filtering, etc.).  
- **Data Management** scripts to read/write `.xdf`, `.edf`, and custom data formats.  
- **Visualization** scripts to analyze and plot results.  

The goal is to help researchers or students replicate an **EEG/EMG-based cognitive load study**, or adapt the tasks to their own experiments.

---

## Repository Structure

Cognitive_Load/ ├─ archives/ ├─ data/ ├─ docs/ ├─ intanutil/ ├─ resources/ ├─ src/ │ ├─ core/ │ ├─ data_management/ │ ├─ experiment_modules/ │ ├─ signal_processing/ │ └─ visualization/ ├─ .gitignore └─ README.md (this file)


- **archives/**  
  Old or experimental scripts that may no longer be actively used (e.g., `analysis_func_blinking.py`, `experiment_waldo.py`, etc.).

- **data/**  
  Contains participant data folders, `.xdf`/`.edf` files, CSV logs, and possibly images/plots. (Ignored by Git if large.)

- **docs/**  
  Documentation or additional reference materials.

- **intanutil/**  
  Utility functions (e.g., `load_intan_rhd_format.py`, `notch_filter.py`) for handling Intan-based data acquisition.

- **resources/**  
  Assets such as background sounds (`back_sound/`) or maze JSONs (`ready_maze/`) for the tasks.

- **src/**  
  Main source code, divided into logical subfolders:
  - **core/**:
    - `main.py` – Potential main entry point.  
    - `config.py`, `utils.py`, `ExperimentState.py` – Core classes/utilities for running experiments.  
    - **calibration/** – Scripts like `jaw_calibration.py`, `white_dot_calibration.py`.  
    - **screens/** – Code for instructions, NASA-TLX, and welcome screens.  
  - **data_management/** – Scripts for reading/writing data formats (`read_xdf.py`, `DataObj.py`).  
  - **experiment_modules/** – Specific tasks like `create_maze.py`, `jaw_movement_task.py`.  
  - **signal_processing/** – Preprocessing & analysis scripts (`edf_reader.py`, `emd_blinking.py`, `jaw_stiffness.py`, etc.). Subfolder `classifying_ica_components/` for automatic ICA component classification.  
  - **visualization/** – Visualization scripts (`vis_cog.py`), images (`face_ica.jpeg`), etc.

---

## Installation & Requirements

1. **Python Version**  
   Recommended: **Python 3.10**.

2. **Create & Activate a Virtual Environment** (optional but recommended):
   ```bash
   # Using conda
   conda create -n cog_load python=3.10
   conda activate cog_load

   # or using venv
   python -m venv venv
   source venv/bin/activate  # (Windows: venv\Scripts\activate)
   
## Data
1. **Running the Main Experiment**  
   To start the main experiment, run:
   ```bash
   python src/core/main.py
   ```
   It may launch an interactive experiment with maze/n-back tasks or calibration flows (depending on your settings in `config.py` or `ExperimentState.py`).
python src/core/main.py
2.  **Calibration Modules**