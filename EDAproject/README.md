# ECP2 AP1 General

## Overview

This is the project with the EDA univariable and multivariable.
<br>
 This README provides an overview of the project, setup instructions, and key functionalities.


## Getting Started

### Prerequisites

- Python 3.10
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Jorgesd13/AP1.git
   ```
   
2. Create a "creds/credentials.yml" file with the following settings for database connection
```
odbc_string: 'DSN=...;Database=...;TrustedConnection=...;MARS_Connection=...;UID=...;PWD=...'
```

In this case, we take a pickle file with the serialized DF as the credentials.yml was for connection with the company´s DB.
   
3. Add submodule DigiPythonTools:
   ```bash
   git submodule add https://github.com/Jorgesd13/DigiPythonTools.git
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Code Structure

The project structure is organized as follows:

- `EDAMV.ipynb`: jupyter NB with data prior to outage.
- `dfXY_coils.pickle`: A DF containing coil IDs column of the dfXY DF.
- `dfXY.pickle`: DF with project data.
- `dfXYFull.pickle`: dfXY with extra variables.
- `EDA1V`: Folder with EDA1V and plotly EDA Dash plots.
- `EDA1V_new.ipynb`: jupyter NB with pre-processing for the EDA1V.
- `EDA1VAirGas_dash_new.ipynb`: jupyter NB with EDA1V dash plots.
- `EDA1VAirGas_new.ipynb`: jupyter NB with EDA1V dash variables.
- `EDA1VrestVars_new.ipynb`: jupyter NB with EDA1V with the rest of variables.
- `EDA1V_new.py`: python file with pre-processing for the EDA1V.
- `multiprocess_try.py`: pyhton file with use example of multiprocess lib.
- `multiprocessing_method.py`: python file with multiprocessing library for plotly creation.
- `images`: Folder with EDA plots. 

