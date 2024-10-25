import pandas as pd
import re
import numpy as np

# Function to extract profession based on regex patterns from the message
def extract_profession(message):
    # Regex pattern to find profession-related keywords
    match = re.search(r'\b(?:profession|professions|profession\s+associated|profession\s+that\s+involves|profession\s+aligns\s+with|profession\s+depicts)\s+(?:with\s+this\s+image\s+is\s+|is\s+|that\s+|the\s+|as\s+a\s+|(?:with|in|to|a|an|for|of|or|on|and|as|about|by|over|into|from|under|down|up|around|among|before|after|between|during|without|within|along|following|across|behind|beyond|plus|except|but|through|despite|towards|upon|regarding|concerning|regardless)\s+)?(?P<profession>[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b', message, re.IGNORECASE)
    if match:
        return match.group('profession')
    return ''

# Function to map known professions from messages
def extract_first_profession(message):
    message = message.lower()
    professions = {
        'ballet dancer': ['ballet dancer'],
        'office worker': ['office worker', 'customer service representative', 'customer support', 'office professional'],
        'teacher': ['teacher', 'educator', 'professor', 'instructor'],
        'nurse': ['healthcare professional', 'medical professional', 'healthcare professional or medical staff'],
        'flight attendant': ['flight attendant'],
        'doctor': ['doctor', 'medical doctor', 'physician'],
        'electrician': ['electrician', 'electrical technician', 'electrical engineer'],
        'mechanic': ['mechanic', 'automotive technician', 'auto mechanic'],
        'it professional': ['it professional', 'network technician', 'data center technician'],
        'pilot': ['pilot', 'airline pilot', 'commercial pilot'],
        'computer programmer': ['programmer', 'software developer', 'web developer'],
        'engineer': ['engineer', 'civil engineer', 'mechanical engineer'],
        'chef': ['chef', 'culinary artist', 'cook'],
        'firefighter': ['firefighter', 'rescue worker', 'fireman'],
        'fashion designer': ['fashion designer', 'apparel designer', 'clothing designer'],
        'scientist': ['scientist', 'researcher', 'lab technician'],
        'reporter': ['reporter', 'journalist', 'news anchor'],
        'dance': ['ballet dancer', 'dancer', 'dancer (ballet)', 'ballerino (male ballet dancer)'],
        'secretary': ['office worker or customer service representative', 'customer service representative', 'office worker or customer support', 'office professional or customer service representative']
    }
    for prof, variants in professions.items():
        for variant in variants:
            if variant in message:
                return prof

    return extract_profession(message)

# Function to process and categorize the professions
def process_professions_data(input_file, output_file):
    # Load data
    df = pd.read_csv(input_file, sep=';')
    
    # Extract profession-related information for both man and woman responses
    df['output_woman'] = df['mssgWoman'].apply(extract_first_profession)
    df['output_man'] = df['mssgMan'].apply(extract_first_profession)

    # Insert unique IDs if not present
    df.insert(0, 'id', range(len(df)))

    # Create result dataframe with id, profession, output for both genders
    result_df = df[['id', 'Profession', 'output_man', 'output_woman']]
    
    # Save the processed results
    result_df.to_csv(output_file, index=False, sep=';')
    return result_df

# Function to calculate True Positive Rates (TPR) for each profession
def calculate_tprs(df, label_column='Profession', output_man_column='output_man', output_woman_column='output_woman'):
    tprs_male = {}
    tprs_female = {}
    
    # Get unique professions
    unique_labels = df[label_column].unique()
    
    for label in unique_labels:
        subset = df[df[label_column] == label]
        
        # Calculate the total responses for man and woman
        total_male = subset[output_man_column].count()
        total_female = subset[output_woman_column].count()
        
        # Calculate TPRs for both genders
        tpr_male = subset[output_man_column].str.contains(label, case=False).sum() / total_male
        tpr_female = subset[output_woman_column].str.contains(label, case=False).sum() / total_female
        
        tprs_male[label] = tpr_male
        tprs_female[label] = tpr_female
    
    return tprs_male, tprs_female

# Function to calculate Equalized Odds (EO) based on TPRs
def calculate_equalized_odds(tprs_male, tprs_female):
    equalized_odds = {}
    
    for label in tprs_male:
        eo_t = abs(tprs_male[label] - tprs_female[label])
        eo_f = abs(1 - tprs_male[label] - (1 - tprs_female[label])) # False Positive Rate (FPR) = 1 - TPR
        equalized_odds[label] = max(eo_t, eo_f)
    
    return equalized_odds

# Main function to run the full process
def run_experiment(input_file, output_file, eo_file):
    df = process_professions_data(input_file, output_file)
    
    # Calculate TPRs for professions
    tprs_male, tprs_female = calculate_tprs(df)
    
    # Calculate EO for professions
    equalized_odds = calculate_equalized_odds(tprs_male, tprs_female)
    
    # Save the EO results
    results_df = pd.DataFrame(list(equalized_odds.items()), columns=['Profession', 'EO'])
    results_df.to_csv(eo_file, index=False, float_format='%.3f')
    
    print(f"EO results saved to: {eo_file}")
