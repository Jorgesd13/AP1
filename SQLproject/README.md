# ECP2 AP1 General

## Overview

This is the project with the ETL.
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

The code provided in this project is merely a guide of how to perform the ETL step, as DB access is only available through the company servers. 


## Code Structure

The project structure is organized as follows:

- `DigiSQLStartup.ipynb`: jupyter NB with the code to access the company's DB. 
- `sql_new.ipynb`: jupyter NB with the ETL step. 



