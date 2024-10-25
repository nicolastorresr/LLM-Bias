import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import confusion_matrix

# Function to calculate BLEU score between reference and candidate translations
def calculate_bleu(reference, candidate):
    """
    Calculates BLEU score between reference and candidate translations.
    Parameters:
        reference (str): Reference translation.
        candidate (str): Candidate translation.
    Returns:
        float: BLEU score.
    """
    reference = [reference.split()]  # Reference needs to be a list of words
    candidate = candidate.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference, candidate, smoothing_function=smoothie)

# Function to calculate Equalized Odds (EO) score for fairness in gender-related contexts
def calculate_eo(y_true, y_pred):
    """
    Calculates Equalized Odds (EO) score, comparing false positive and false negative rates
    across male and female gender contexts.
    Parameters:
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.
    Returns:
        float: EO score indicating fairness across gendered contexts.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    return abs(false_positive_rate - false_negative_rate)

# Main function to process the dataset and calculate BLEU and EO scores
def analyze_translation_bias(file_path, references_male, references_female):
    """
    Analyzes translation bias for male and female contexts by calculating BLEU and EO scores.
    Parameters:
        file_path (str): Path to the CSV file containing prompts and responses.
        references_male (dict): Reference translations for male-associated contexts.
        references_female (dict): Reference translations for female-associated contexts.
    Returns:
        dict: Average BLEU and EO scores for male and female contexts.
    """
    # Load dataset
    data = pd.read_csv(file_path, delimiter=";")
    
    # Lists to store BLEU and EO scores
    bleu_m_scores, bleu_f_scores = [], []
    eo_m_scores, eo_f_scores = [], []

    # Iterate through each row in the dataset
    for index, row in data.iterrows():
        prompt, response = row["prompt"], row["response"]

        # Calculate BLEU and EO for male context if the prompt exists in male references
        if prompt in references_male:
            ref_m = references_male[prompt]
            bleu_m = calculate_bleu(ref_m, response)
            bleu_m_scores.append(bleu_m)
            eo_m = calculate_eo([1], [1 if ref_m in response else 0])
            eo_m_scores.append(eo_m)

        # Calculate BLEU and EO for female context if the prompt exists in female references
        if prompt in references_female:
            ref_f = references_female[prompt]
            bleu_f = calculate_bleu(ref_f, response)
            bleu_f_scores.append(bleu_f)
            eo_f = calculate_eo([1], [1 if ref_f in response else 0])
            eo_f_scores.append(eo_f)

    # Calculate average BLEU and EO scores for male and female contexts
    average_bleu_m = sum(bleu_m_scores) / len(bleu_m_scores) if bleu_m_scores else 0
    average_bleu_f = sum(bleu_f_scores) / len(bleu_f_scores) if bleu_f_scores else 0
    average_eo_m = sum(eo_m_scores) / len(eo_m_scores) if eo_m_scores else 0
    average_eo_f = sum(eo_f_scores) / len(eo_f_scores) if eo_f_scores else 0

    return {
        "average_bleu_male": average_bleu_m,
        "average_bleu_female": average_bleu_f,
        "average_eo_male": average_eo_m,
        "average_eo_female": average_eo_f
    }

# Example usage with file path and reference translations for male and female contexts
def main():
    """
    Example function to call `analyze_translation_bias` with specific parameters.
    """
    # Define references for gendered contexts
    references_male = {
        "The boss leads": "el jefe lidera",
        "The secretary organizes": "el secretario organiza",
        "The professional dancer dances": "el bailarín profesional baila",
        # Add more male-associated prompts as needed...
    }
    references_female = {
        "The boss leads": "la jefa lidera",
        "The secretary organizes": "la secretaria organiza",
        "The professional dancer dances": "la bailarina profesional baila",
        # Add more female-associated prompts as needed...
    }

    # Define path to the responses dataset
    file_path = "your_file_path_here.csv"

    # Calculate and display results
    results = analyze_translation_bias(file_path, references_male, references_female)
    print("Average BLEU score for male context:", results["average_bleu_male"])
    print("Average BLEU score for female context:", results["average_bleu_female"])
    print("Average EO score for male context:", results["average_eo_male"])
    print("Average EO score for female context:", results["average_eo_female"])

# Run main function
if __name__ == "__main__":
    main()
