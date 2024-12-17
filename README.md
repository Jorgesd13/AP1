# ECP2 AP1 General

## Overview

This is the project with the ETL, EDA, modelling and dashboards.
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
 


## Code Structure
The project structure is organized as follows:

- `DASHproject`: folder with files required to create the dashboards. 
- `MODELproject`: folder with files required to train and compare the different models as well as compute the statistical value of temp SPs..
- `EDAproject`: folder with files required to perform the MV and 1V EDA. 
- `ETLproject`: folder with files required to perform the ETL step. 



