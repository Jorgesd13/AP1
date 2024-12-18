# ECP2_MaintenanceDashboard

## Overview

This is the Maintenance dashboard Pipeline.
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

### Running the Application

Execute the following command in the terminal to create the dashboard:

```bash
python MaintenanceDashboard.py
```


## Code Structure

The project structure is organized as follows:

- `MaintenanceDashboard.py`: Maintenance dashboard file.
- `DashPreprocessing.py`: Preprocessing file (default queries 5 days). As it queries the unavailable company DB, in this example it loads the dfXY.pickle file
- `assets`: Folder with CSS stylesheets
- `DigiPythonTools`: Custom utility module.

