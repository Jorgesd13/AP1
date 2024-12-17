# ECP2 AP1 Model Pipeline

## Overview

This project cointains the Pipeline built in Python to retrain the regression model for the pyro2 prediction of the AP1 line, as well as the jupyter NBs used to train all models and some statistical calculations.
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

Execute the following command in the terminal to train the model:

```bash
python ModelTraining.py
```


## Code Structure

The project structure is organized as follows:

- `ModelTraining.py`: Train Model (Histogram Gradient Boosting Descend Regressor or Gradient Boosting Descend Regressor depending on number of observations and hyperparameters of best Tree Regressor) file.
- `ModelPreprocessing.py`: Preprocessing file (defaults queries 5 days).
- `AllModels`: Folder with jupyter NB with code for training all models and .pickle files with trained models.
- `ZoneTempSP_statistical`: Folder with jupyter NB with code for computing statistical furnace temp SP and img files with statistical temp SPs results.
- `DigiPythonTools`: Custom utility module.
- `.pickle files`: GridSearchs for final model hyperparam and Trained model

