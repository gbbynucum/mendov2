import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset from CSV file with error handling
try:
    medicines_df = pd.read_csv(r'C:\Users\Geb\Documents\VSC Projects\Mendov2\symptom_medication.csv', on_bad_lines='skip')  # Adjust parameters as needed
except pd.errors.ParserError as e:
    print(f"Error reading the CSV file: {e}")
    exit()

# Load maintenance medicine data
try:
    maintenance_df = pd.read_csv(r'C:\Users\Geb\Documents\VSC Projects\Mendov2\maintenance.csv', on_bad_lines='skip')
except pd.errors.ParserError as e:
    print(f"Error reading the maintenance CSV file: {e}")
    exit()

# Print the DataFrame to check its structure
print(medicines_df.head())  # Check the first few rows
print(medicines_df.columns.tolist())  # Print column names to check for discrepancies

# Check if 'symptom' column exists
if 'symptom' not in medicines_df.columns:
    print("Error: 'symptom' column not found in the DataFrame.")
    exit()

# Define synonyms to enhance symptom matching
synonyms = {
    'headache': ['head pain', 'migraine', 'labad ulo'],
    'fever': ['high temperature', 'lagnat'],
    'cough': ['coughing', 'hack', 'ubo'],
    'nausea': ['sick', 'queasy'],
    'stomach ache': ['sakit tiyan', 'sakit sa tiyan', 'pagsakit ng tiyan'],  # Added synonyms for stomach ache
    # Add more synonyms as needed
}

# Function to expand symptoms using synonyms
def expand_symptoms(text):
    for key, values in synonyms.items():
        for synonym in values:
            text = re.sub(r'\b' + re.escape(synonym) + r'\b', key, text, flags=re.IGNORECASE)
    return text

# Preprocessing function
def preprocess_text(text):
    # Expand synonyms
    text = expand_symptoms(text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenization
    words = nltk.word_tokenize(text.lower())
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Create a copy of the original DataFrame for evaluation purposes
original_df = medicines_df.copy()

# Combine symptoms into a single string for each medication
medicines_df['processed_symptoms'] = medicines_df.groupby('medication')['symptom'].transform(lambda x: ' '.join(x))
medicines_df = medicines_df[['medication', 'processed_symptoms', 'alternative_medication']].drop_duplicates()

# Fill NaN values with an empty string
medicines_df['processed_symptoms'] = medicines_df['processed_symptoms'].fillna('')

# Apply preprocessing to the dataset
medicines_df['processed_symptoms'] = medicines_df['processed_symptoms'].apply(preprocess_text)

# Vectorization
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(medicines_df['processed_symptoms'])

# Function to get avoid list for the entered maintenance medicines
def get_avoid_list(maintenance_medicines):
    avoid_list = []
    for medicine in maintenance_medicines:
        # Normalize the medicine name for comparison
        normalized_medicine = medicine.strip().lower()
        # Find all medications to avoid for the given maintenance medicine
        avoid_list.extend(maintenance_df[maintenance_df['Maintenance'].str.lower() == normalized_medicine]['Avoid'].dropna().tolist())
    return list(set(avoid_list))  # Remove duplicates

# Function to get the list of allergies from the user
def get_allergies():
    allergies_input = input("Enter any allergies you have (comma-separated): ")
    return [allergy.strip().lower() for allergy in allergies_input.split(',')]


# Function to recommend medicines based on input symptoms, maintenance, and allergies
def recommend_medicine(input_symptoms, maintenance_medicines, user_allergies):
    avoid_list = get_avoid_list(maintenance_medicines)
    
    # Split the input symptoms into a list and preprocess each symptom
    symptoms_list = [symptom.strip() for symptom in input_symptoms.split(',')]
    
    # Initialize the list to hold all recommendations
    all_recommendations = []  # Initialize the list here
    
    for symptom in symptoms_list:
        processed_symptom = preprocess_text(symptom)
        input_vector = tfidf.transform([processed_symptom])
        
        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
        
        # Get the top 3 recommendations
        recommended_indices = cosine_similarities.argsort()[-3:][::-1]
        
        symptom_recommendations = set()  # Use a set to avoid duplicates
        all_valid_alternatives = set()  # Set to collect all valid alternatives
        
        for index in recommended_indices:
            med = medicines_df.iloc[index]['medication']
            alternatives = medicines_df.iloc[index]['alternative_medication']
            
            # Check if the recommended medicine is in the avoid list or user allergies
            if med.lower() not in [avoid.lower() for avoid in avoid_list] and med.lower() not in user_allergies:
                symptom_recommendations.add(f"Medicine recommended for {symptom}: {med}")
                
                # Check alternatives against the avoid list and user allergies
                if pd.notna(alternatives) and alternatives:
                    alternative_list = [alt.strip() for alt in alternatives.split(';')]
                    valid_alternatives = [alt for alt in alternative_list if alt and alt.lower() not in [avoid.lower() for avoid in avoid_list] and alt.lower() not in user_allergies]
                    
                    # Add valid alternatives to the set
                    all_valid_alternatives.update(valid_alternatives)
        
        # If no direct medication was found, check for alternatives
        if not symptom_recommendations:
            if pd.notna(alternatives) and alternatives:
                alternative_list = [alt.strip() for alt in alternatives.split(';')]
                valid_alternatives = [alt for alt in alternative_list if alt and alt.lower() not in [avoid.lower() for avoid in avoid_list] and alt.lower() not in user_allergies]
                
                if valid_alternatives:
                    all_valid_alternatives.update(valid_alternatives)
                else:
                    symptom_recommendations.add(f"No suitable medication found for {symptom}. Please consult a healthcare provider.")
            else:
                symptom_recommendations.add(f"No suitable medication found for {symptom}. Please consult a healthcare provider.")
        
        # Add all valid alternatives to the recommendations
        if all_valid_alternatives:
            symptom_recommendations.add(f"  Alternatives for {symptom}: {', '.join(sorted(all_valid_alternatives))}")
        
        all_recommendations.extend(symptom_recommendations)  # Add the current symptom's recommendations to the overall list

    # Remove redundancy by converting to a set and back to a list
    return list(set(all_recommendations))

# Function to evaluate the model using the original DataFrame
def evaluate_model(original_df):
    correct_predictions = 0
    total_predictions = 0

    # Iterate through the original dataset to create test cases
    for index, row in original_df.iterrows():
        symptoms = row['symptom']
        correct_med = row['medication']
        
        # Pass an empty list for allergies during evaluation
        recommendations = recommend_medicine(symptoms, [correct_med], [])  # Pass an empty list for allergies
        
        # Check if the recommended medications match the correct medications
        if any(correct_med in rec for rec in recommendations):
            correct_predictions += 1
        total_predictions += 1

    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        loss = 1 - accuracy  # Loss is the proportion of incorrect predictions
        print(f"Accuracy of the model: {accuracy * 100:.2f}%")
        print(f"Loss of the model: {loss * 100:.2f}%")
    else:
        print("No predictions to evaluate.")

# Loop for multiple iterations
def main():
    # Evaluate the model immediately after loading the dataset
    print("Evaluating the model with the current dataset...")
    evaluate_model(original_df)

    while True:
        # Get user input for symptoms and maintenance medicines
        input_symptoms = input("Enter your symptoms (comma-separated): ")
        maintenance_medicines = input("Enter your maintenance medicines (comma-separated): ").split(',')
        
        # Get user allergies
        user_allergies = get_allergies()

        # Call the recommendation function with user input
        recommendations = recommend_medicine(input_symptoms, maintenance_medicines, user_allergies)

        # Print the recommendations
        for recommendation in recommendations:
            print(recommendation)

        # Evaluate the model after each user input
        print("Evaluating the model after your input...")
        evaluate_model(original_df)

        # Ask if the user wants to continue
        continue_prompt = input("Do you want to enter more symptoms? (yes/no): ")
        if continue_prompt.lower() != 'yes':
            break

if __name__ == "__main__":
    main()
