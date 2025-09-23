# Big Data Hackathon - Weekly Sales Forecasting Model

This project contains the notebooks used to train the models for the Big Data hackathon held from 09/09 to 22/09. The best model overall (speed, wmape, etc.) was a LightGBM model which was trained on a T4 GPU with high RAM on Colab (train_lgbm_capped.ipynb).


## Setup for test data

Before running the scripts, set up the required environment.

1.  **Clone the repository** (or download the files).

    ```bash
    git clone https://github.com/itsalissonsilva/ds_bigdata_hackathon
    ```

2.  **Install the dependencies** by running the following command in your terminal from the project's root directory:
   
    ```bash
    pip install -r requirements.txt
    ```


## Adding new data


### Step 1: Add Your Data

We have included a folder named jan23_simul created from the train data in order to check if the scripts to preprocess the data and evaluate the model work correctly. You can run preprocessing.py and evaluate.py directly or modify the path source to your actual January data in the script itself by following the instructions below. 

---
1.  Extract your data to a folder in the project's directory.
2.  Unzip your new raw data and place the three Parquet files (e.g., `part27.snappy.parquet`, `part51.snappy.parquet`, `part71.snappy.parquet`) inside a folder you'll determine in the path, it can be the data folder.
3. Update the info in the preprocessing script to use the same path you unzipped it to, change the file names too to the ones in your zipped file:

```bash
    IN_DIR = BASE_DIR / "data" / "jan23_simul"
    df27 = pd.read_parquet(IN_DIR / "part27_jan.parquet")
    df71 = pd.read_parquet(IN_DIR / "part71_jan.parquet")
    df51 = pd.read_parquet(IN_DIR / "part51_jan.parquet")
```

The data must have the same format as the train data provided in order to work correctly.

---

### Step 2: Run the Preprocessing Script

This script will find the raw Parquet files in your folder, process them, and save the result.

In your terminal, run:
```bash
python preprocess.py
```
This will create a single, model-ready test.csv file inside the data/ folder.

---
### Step 3: Run the Evaluation Script

This script uses the test.csv file you just created to score the model. Unzip the trained model from the models folder and modify the paths accordingly:

```bash
BASE_DIR   = Path(__file__).resolve().parent
TEST_CSV   = BASE_DIR / "test.csv"
MODEL_PATH = BASE_DIR / "lgbm_model_cap.pkl"              # unzip the model first
OUT_PREDS  = BASE_DIR / "data" / "predictions.csv"
```


then in your terminal, run:

```bash
python evaluate.py
```

This will return the wmape for the model on that period.




## Models tested

Exploratory Data Analysis - Hevenicio  
Linear Regression Baseline - Gabriel  
Extreme Gradient Boosting (XGB) - João  
Light Gradient Boosting (LGBM) - Alisson  
LSTM (w/ Keras) - Clara  
LSTM (w/ Pytorch) - Alisson  




## Improvements tbd

* Database support
* API endpoint
* train script to update model in case of drift
* forecast.py
* AWS Deployment (or any other Cloud provider)
* Clarification regarding negative values etc
* Grid search more robust
* Test more SOTA models: TimesFM, Neural Prophet, TFT etc. 
