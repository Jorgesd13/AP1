# ECP2 AP1 General

## Overview

This is the general project with the ETL, EDA and firsts models and dash attempts.
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
   
3. Add submodule DigiPythonTools:
   ```bash
   git submodule add https://github.com/Jorgesd13/DigiPythonTools.git
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Considerations

Due to an update in the furnace after an outage, there's a folder called CleanData with data after the outage.


Jupyter Notebook codes link :[ http://10.1.16.189:3050/dev/lab/tree/coldmill/js5296/AP1](http://10.1.16.189:3050/dev/lab/tree/coldmill/js5296/AP1)


## Code Structure

The project structure is organized as follows:

- `Alvero`: Folder with Classification/Regression models of other projects to use as skeleton.
- `CleanData`: Folder with ETL, EDA1V and EDAMV with new data after outage.
- `Dash`: Folder Plotly and Dash files, early version of the Maintenance Dashboard Pipeline.
- `EDA1V`: Folder with EDA1V using data before outage (old data).
- `modeling`: Folder with models' ETL, Preprocessing, trained regression models and Zone Temp SP statistical computation.
- `EDAMV.ipynb`: jupyter NB with data prior to outage.
- `EDAMV_Vold.ipynb`: Old version of EDAMV.ipynb with data prior to outage.

