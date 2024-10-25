import pandas as pd
import openai

# Add your OpenAI API key here
openai.api_key = "Insert_your_api_key_here"

# Define function to call the GPT-4 model with the specified prompt and token limit
def gpt4o(prompt, max_tokens=300):
    # Generate response from the GPT-4 model
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
    )
    # Return the generated text content
    return response['choices'][0]['message']['content'].strip()

# Function to duplicate rows in a DataFrame, used for generating multiple narrative variations
def duplicate_rows(df, times=10):
    # Repeat each row 'times' times to create multiple narrative prompts
    duplicated_df = pd.DataFrame(df.loc[df.index.repeat(times)].reset_index(drop=True))
    return duplicated_df

# Function to process each prompt, generate response, and save to a CSV
def process_prompts_from_csv(df, csv_out):
    # Extract the prompts from the first column
    prompts = df.iloc[:, 0]
    responses = []

    # Generate a response for each prompt
    for prompt in prompts:
        response = gpt4o(prompt)
        responses.append(response)

    # Save the responses in the DataFrame and export as CSV
    df['response'] = responses
    df.to_csv(csv_out, index=False, sep=";", quoting=1, encoding="UTF-8")
    print(f'Responses have been saved to "{csv_out}"')

# Main function to read prompts, duplicate them for multiple narrative samples, and save responses
def Narrative_Bias_Prompts(csv_in, csv_out):
    # Load the input CSV containing gender-neutral narrative prompts
    original_df = pd.read_csv(csv_in, delimiter=";", quoting=1)

    # Duplicate rows for varied prompt generation per context
    duplicated_df = duplicate_rows(original_df)

    # Process duplicated prompts and save responses
    process_prompts_from_csv(duplicated_df, csv_out)

# Formula to calculate Male-to-Female Pronoun Ratio, based on pronoun counts in narratives
def calculate_mf_ratio(male_count, female_count):
    # Ratio formula as per Zhao et al. (2018), handles division by zero cases
    if female_count == 0:
        return float('inf') if male_count > 0 else 0
    return male_count / female_count

# Formula to calculate the Average Stereotype Score (ASS) for gender-based attributes
def calculate_average_stereotype_score(stereotype_counts, total_attributes):
    # Avoid division by zero; if no attributes found, return 0 for unbiased
    if total_attributes == 0:
        return 0
    return sum(stereotype_counts) / total_attributes
