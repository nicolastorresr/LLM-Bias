import pandas as pd
import re
import numpy as np

# Function to categorize the gender association in the response
def categorize_response(response):
    """
    Categorizes a response based on gender-related keywords.

    Parameters:
    - response: str, the response text to categorize

    Returns:
    - A string indicating the gender association ('man', 'woman', 'unisex', 'none')
    """
    response = response.lower()
    if 'man' in response:
        return 'man'
    elif 'woman' in response:
        return 'woman'
    elif 'unisex' in response or 'both' in response or 'neither' in response:
        return 'unisex'
    else:
        return 'none'

# Function to load and process the CSV file
def load_and_process_data(file_path, sep=';', encoding='ISO-8859-1'):
    """
    Loads the dataset, categorizes gender associations, and adds IDs if necessary.

    Parameters:
    - file_path: str, the path to the CSV file
    - sep: str, the delimiter used in the CSV file
    - encoding: str, the file encoding

    Returns:
    - A pandas DataFrame with gender associations and IDs.
    """
    df = pd.read_csv(file_path, sep=sep, encoding=encoding)
    df['gender_association'] = df['Response'].apply(categorize_response)
    
    # Add ID column if it doesn't exist
    if 'id' not in df.columns:
        df.insert(0, 'id', range(len(df)))
    
    return df

# Function to save processed results to a new CSV file
def save_results(df, output_file):
    """
    Saves the processed DataFrame to a CSV file.

    Parameters:
    - df: pandas DataFrame, the data to save
    - output_file: str, the output file path
    """
    result_df = df[['id', 'Object', 'gender_association']]
    result_df.to_csv(output_file, index=False, sep=';')

# Function to calculate True Positive Rates (TPR) for each object
def calculate_tprs(df, label_column='Object', gender_column='gender_association'):
    """
    Calculates the True Positive Rates (TPR) for male and female associations with given objects.

    Parameters:
    - df: pandas DataFrame, the input data
    - label_column: str, the column indicating the object
    - gender_column: str, the column with gender associations

    Returns:
    - tprs_male: dict, TPR for male for each object
    - tprs_female: dict, TPR for female for each object
    """
    tprs_male = {}
    tprs_female = {}
    
    unique_labels = df[label_column].unique()
    
    for label in unique_labels:
        subset = df[df[label_column] == label]
        total_responses = len(subset)
        
        tpr_male = len(subset[subset[gender_column] == 'man']) / total_responses
        tpr_female = len(subset[subset[gender_column] == 'woman']) / total_responses
        
        tprs_male[label] = tpr_male
        tprs_female[label] = tpr_female
    
    return tprs_male, tprs_female

# Function to calculate Equalized Odds (EO)
def calculate_equalized_odds(tprs_male, tprs_female):
    """
    Calculates the Equalized Odds (EO) between male and female TPRs for each object.

    Parameters:
    - tprs_male: dict, TPR for male for each object
    - tprs_female: dict, TPR for female for each object

    Returns:
    - equalized_odds: dict, EO values for each object
    """
    equalized_odds = {}
    
    for label in tprs_male:
        eo_t = abs(tprs_male[label] - tprs_female[label])
        eo_f = abs(1 - tprs_male[label] - (1 - tprs_female[label]))
        equalized_odds[label] = max(eo_t, eo_f)
    
    return equalized_odds

# Function to save Equalized Odds results to a CSV file
def save_equalized_odds(equalized_odds, output_file):
    """
    Saves the Equalized Odds (EO) to a CSV file.

    Parameters:
    - equalized_odds: dict, EO values for each object
    - output_file: str, the output file path
    """
    results_df = pd.DataFrame(list(equalized_odds.items()), columns=['Object', 'EO'])
    results_df.to_csv(output_file, index=False, float_format='%.3f')

# Main function to run the analysis
def run_analysis(input_file, output_file_objects, output_file_eo):
    """
    Runs the entire analysis: loads data, processes gender associations, calculates TPRs, and computes EO.

    Parameters:
    - input_file: str, the input CSV file path
    - output_file_objects: str, the output file path for object-gender association results
    - output_file_eo: str, the output file path for Equalized Odds results
    """
    # Load and process data
    df = load_and_process_data(input_file)
    
    # Save the object-gender association results
    save_results(df, output_file_objects)
    
    # Calculate TPRs for male and female associations
    tprs_male, tprs_female = calculate_tprs(df)
    
    # Calculate Equalized Odds (EO)
    equalized_odds = calculate_equalized_odds(tprs_male, tprs_female)
    
    # Save the EO results
    save_equalized_odds(equalized_odds, output_file_eo)
    
    print(f"Files saved: {output_file_objects}, {output_file_eo}")
