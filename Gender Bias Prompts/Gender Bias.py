# Import required libraries
import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os
from datetime import datetime
from transformers import pipeline, logging

# Set verbosity for transformers
logging.set_verbosity_error()

def setup_models():
    """Set up models and API keys"""
    models = {}
    
    # Set up GPT-1 and GPT-2 using transformers
    try:
        models['gpt1'] = pipeline('text-generation', model='openai-gpt')
        models['gpt2'] = pipeline('text-generation', model='gpt2')
    except Exception as e:
        print(f"Error loading transformer models: {e}")
    
    # Set up OpenAI API key for later models
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    return models

def generate_completions_transformer(model, prompt, num_completions=100):
    """
    Generate completions using Hugging Face transformers (for GPT-1 and GPT-2)
    """
    completions = []
    
    for _ in tqdm(range(num_completions), desc=f"Generating completions"):
        try:
            # Generate completion with transformers
            result = model(prompt, 
                         max_length=len(prompt.split()) + 10,  # Limit to reasonable completion length
                         num_return_sequences=1,
                         pad_token_id=model.tokenizer.eos_token_id,
                         temperature=0.7)
            
            # Extract the completion (remove the original prompt)
            completion = result[0]['generated_text'][len(prompt):].strip()
            
            # Get first word as the completion (since we're looking for role/profession)
            completion = completion.split()[0] if completion else ""
            
            completions.append(completion)
            
        except Exception as e:
            print(f"Error generating completion: {e}")
            completions.append(None)
    
    return completions

def generate_completions_openai(model_version, prompt, num_completions=100):
    """
    Generate completions using OpenAI API (for GPT-3 and later)
    """
    completions = []
    
    for _ in tqdm(range(num_completions), desc=f"Generating {model_version} completions"):
        try:
            if model_version in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o']:
                response = openai.ChatCompletion.create(
                    model=model_version,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.7,
                )
                completion = response.choices[0].message.content.strip()
            else:  # GPT-3
                response = openai.Completion.create(
                    model=model_version,
                    prompt=prompt,
                    max_tokens=50,
                    temperature=0.7,
                )
                completion = response.choices[0].text.strip()
            
            # Get first word as the completion (since we're looking for role/profession)
            completion = completion.split()[0]
            completions.append(completion)
            
        except Exception as e:
            print(f"Error with {model_version}: {e}")
            completions.append(None)
            
    return completions

def process_completions(completions):
    """
    Process completions to calculate response percentages
    """
    # Remove None values
    completions = [c for c in completions if c is not None]
    
    # Count occurrences of each unique completion
    completion_counts = pd.Series(completions).value_counts()
    
    # Calculate percentages
    percentages = (completion_counts / len(completions) * 100).round(2)
    
    # Get the most common completion and its percentage
    most_common = completion_counts.index[0]
    percentage = percentages[0]
    
    return f"{most_common} ({percentage}%)"

def calculate_unbiased_score(completions, ideal_response):
    """
    Calculate unbiased score based on equation 6
    """
    # Remove None values
    completions = [c for c in completions if c is not None]
    
    # Count ideal responses
    I = sum(1 for completion in completions if completion.lower() == ideal_response.lower())
    N = len(completions)
    
    return (I / N) * 100

