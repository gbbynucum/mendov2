from flask import Flask, render_template, request, redirect, make_response, url_for, session, flash
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import uuid


# Download necessary NLTK resources
nltk.download('punkt_tab')  
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = '27a4c9a0c0c522b2b26eb6ea3d1acc12'
# Load dataset from CSV file with error handling
try:
    medicines_df = pd.read_csv(r'C:\Users\Geb\Documents\VSC Projects\mendov2\symptom_medication.csv', on_bad_lines='skip')
except pd.errors.ParserError as e:
    print(f"Error reading the CSV file: {e}")
    exit()

# Load maintenance medicine data
try:
    maintenance_df = pd.read_csv(r'C:\Users\Geb\Documents\VSC Projects\mendov2\maintenance.csv', on_bad_lines='skip')
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
    'headache': [
        'head pain', 'migraine', 'labad ulo', 'sakit akong ulo', 'my head hurts', 'my head is painful', 'sakit ulo', 'sakit akong ulo', 'sakit kaayo akong ulo'
    ],
    'fever': [
        'high temperature', 'lagnat', 'gilagnat', 'gilagnat ko', 'bugat ang lawas', 'bugat ang paminaw', 'hilanat',
    ],
    'cough': [
        'coughing', 'hack', 'ubo', 'ubo-ubo', 'nauubo'
    ],
    'sore throat': [
        'sakit akong tilaok', 'nagasakit akong tutunlan', 'gasakit akong tilaok', 'sakit kaayo akong tilaok', 'grabe ka sakit sakong tilaok',
        'sigeg ka sakit akong tilaok', 'sakit ilunok', 'sakit kaayo mulunok'
    ], 
    'insomnia':[
        'cant sleep', 'unable to sleep', 'dili katulog', 'wakefulness', 'sleep disturbance', 'sleeping issues'
        ],
    'nausea': [
        'sick', 'queasy', 'suka', 'kasukaon', 'nasusuka', 'puke',
    ],
    'stomach ache': [
        'sakit tiyan', 'sakit sa tiyan', 'pagsakit ng tiyan'
    ],
    'allergy': [
        'rashes', 'rash', 'skin irritation', 'skin hurts', 'pangatol', 'nangatol', 'katol', 'pagkakaroon ng rashes', 'pangangati', 'makati',
        'katol balat', 'katol akong balat', 'allergies', 'sensitive', 'hypersensitive'
    ],
    'asthma': [
        'hubak', 'hika', 'kahubakon', 'ginahubak', 'gina hubak'
    ],
    'toothache': [
        'sakit ngipon', 'tooth pain', 'toothe paine', 'sakit akong ngipon', 'sakit kaayo akong ngipon', 'ngipon nako sakit kaayo', 'ngipon nako sakit',
        'ngulngul akong ngipon',
    ],
    'menstruation': [
        'gidugo', 'gi dugo', 'dysmenorrhea', 'ginadugo ko', 'mens', 'gi mens ko', 'naa koy menstruation', 'sakit puson', 'puson nako sakit',
        'puson sakit', 'sakit kaayo akong puson', 'sakit dyud kaayo akong puson', 'masakit ang puson ko', 'nagsakit ang puson ko', 'nagasakit ang puson ko',
        'puson masakit', 'red tide', 'red days', 'sumasakit ang puson ko', 'sobrang sakit ng puson ko', 'ang sakit sakit ng puson ko'
    ],
}

# Function to generate a unique cart ID
def generate_cart_id():
    return str(uuid.uuid4())

# Function to get the current cart ID (from cookies or generate a new one)
def get_cart_id():
    cart_id = request.cookies.get('cart_id')
    if not cart_id:
        cart_id = generate_cart_id()
        resp = make_response(redirect('/'))  # Redirect to home page to set the cookie
        resp.set_cookie('cart_id', cart_id)  # Store cart ID in cookies
        return cart_id
    return cart_id

# Function to add an item to the cart in the database
def add_to_cart(medicine, quantity):
    cart_id = get_cart_id()
    
    conn = sqlite3.connect('medicine.db')
    cursor = conn.cursor()
    
    # Check if the item is already in the cart
    cursor.execute("SELECT * FROM carts WHERE cart_id = ? AND medicine_name = ?", (cart_id, medicine[0]))
    existing_item = cursor.fetchone()
    
    if existing_item:
        new_quantity = existing_item[4] + quantity
        cursor.execute("UPDATE carts SET quantity = ? WHERE cart_id = ? AND medicine_name = ?",
                       (new_quantity, cart_id, medicine[0]))
    else:
        cursor.execute("INSERT INTO carts (cart_id, medicine_name, price, quantity, stock) VALUES (?, ?, ?, ?, ?)",
                       (cart_id, medicine[0], medicine[1], quantity, medicine[2]))
    
    conn.commit()
    conn.close()

# Function to remove an item from the cart
def remove_from_cart(medicine_name):
    cart_id = get_cart_id()
    
    conn = sqlite3.connect('medicine.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM carts WHERE cart_id = ? AND medicine_name = ?", (cart_id, medicine_name))
    conn.commit()
    conn.close()

# Function to get all items in the cart
def get_cart_items():
    cart_id = get_cart_id()
    
    conn = sqlite3.connect('medicine.db')
    cursor = conn.cursor()
    cursor.execute("SELECT medicine_name, price, quantity FROM carts WHERE cart_id = ?", (cart_id,))
    items = cursor.fetchall()
    conn.close()
    
    return items

# Function to update item quantity in the cart
def update_cart_quantity(medicine_name, quantity):
    cart_id = get_cart_id()
    
    conn = sqlite3.connect('medicine.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE carts SET quantity = ? WHERE cart_id = ? AND medicine_name = ?",
                   (quantity, cart_id, medicine_name))
    conn.commit()
    conn.close()

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

# Define routes for Flask to render different pages
# Route for the index (home) page
@app.route('/')
def index():
    return render_template('index.html')

# Define the warning route
@app.route('/warning')
def warning():
    return render_template('warning.html')

# Define the disclaimer route
@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')

# Define the welcome route
@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

# Define the shop route
@app.route('/shop', methods=['GET', 'POST'])
def shop():
    # Connect to SQLite database
    conn = sqlite3.connect('medicine.db')
    cursor = conn.cursor()

    # Fetch all medicine data including therapeutic effects and side effects
    cursor.execute("SELECT name, price, stock, image_path, therapeutic_effects, side_effects FROM medicines")
    medicines = cursor.fetchall()

    conn.close()
    return render_template('shop.html', medicines=medicines)

@app.route('/purchase', methods=['POST'])
def purchase():
    # Handle the checkout process here
    # Clear the cart, process payment, etc.
    # For now, redirect to the payment page
    return redirect(url_for('payment'))


@app.route('/cart')
def view_cart():
    cart_items = get_cart_items()  # Get the items from the database
    return render_template('cart.html', cart_items=cart_items)

@app.route('/remove_from_cart/<string:medicine_name>')
def remove_from_cart_route(medicine_name):
    remove_from_cart(medicine_name)  # Remove the item from the cart
    return redirect('/cart')

@app.route('/update_cart', methods=['POST'])
def update_cart():
    medicine_name = request.form.get('medicine_name')
    quantity = int(request.form.get('quantity'))
    update_cart_quantity(medicine_name, quantity)  # Update quantity in the cart
    return redirect('/cart')

@app.route('/payment', methods=['GET', 'POST'])
def payment():
    if request.method == 'POST':
        cart = request.form.get('cart')
        total_amount = float(request.form.get('total_amount', 0))
        inserted_amount = float(request.form.get('inserted_amount', 0))

        # Check if payment is insufficient
        if inserted_amount < total_amount:
            flash(f"Insufficient funds! You need an additional ₱{total_amount - inserted_amount:.2f}.", "error")
            return redirect(url_for('payment'))

        # Process payment and clear cart
        flash("Payment successful! Dispense the medicine.", "success")

        # Return the payment page where the user can dispense the medicine
        return render_template('payment.html', cart=cart, total_amount=total_amount)

    return render_template('payment.html')



# Route for Admin Login Page
@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Connect to the SQLite database
        conn = sqlite3.connect('medicine.db')
        cursor = conn.cursor()

        # Check if the username and password match an entry in the admins table
        cursor.execute("SELECT * FROM admins WHERE username = ? AND password = ?", (username, password))
        admin = cursor.fetchone()
        
        conn.close()

        if admin:
            # Admin is authenticated, store admin info in session
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            # Invalid credentials
            return render_template('admin_login.html', error="Invalid credentials")
    
    return render_template('admin_login.html')

# Protect the admin dashboard route
@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    # Fetch medicine data to display in the admin dashboard
    conn = sqlite3.connect('medicine.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, price, stock FROM medicines")
    medicines = cursor.fetchall()
    conn.close()
    
    # Do not include flash messages unrelated to admin operations
    return render_template('admin_dashboard.html', medicines=medicines)


# Route to add a new medicine
@app.route('/admin/add_medicine', methods=['POST'])
def add_medicine():
    name = request.form['name']
    price = request.form['price']
    stock = request.form['stock']
    image = request.files['image']

    # Save image
    image_filename = None
    if image:
        image_filename = f"static/images/{uuid.uuid4().hex}.jpg"
        image.save(image_filename)

    # Insert into database
    conn = sqlite3.connect('medicine.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO medicines (name, price, stock, image_path) VALUES (?, ?, ?, ?)",
                   (name, price, stock, image_filename))
    conn.commit()
    conn.close()

    flash('Medicine added successfully!', 'success')  # Flash success message
    return redirect(url_for('admin_dashboard'))

# Route to update medicine price
@app.route('/admin/update_price', methods=['POST'])
def update_price():
    medicine_id = request.form.get('medicine_id')
    new_price = request.form.get('price')

    conn = sqlite3.connect('medicine.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE medicines SET price = ? WHERE id = ?", (new_price, medicine_id))
    conn.commit()
    conn.close()

    flash('Price updated successfully!', 'success')  # Flash success message
    return redirect(url_for('admin_dashboard'))


# Route to update medicine stock
@app.route('/admin/update_stock', methods=['POST'])
def update_stock():
    medicine_id = request.form.get('medicine_id')
    new_stock = request.form.get('stock')

    conn = sqlite3.connect('medicine.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE medicines SET stock = ? WHERE id = ?", (new_stock, medicine_id))
    conn.commit()
    conn.close()

    return redirect(url_for('admin_dashboard'))

# Route to delete a medicine
@app.route('/admin/delete_medicine', methods=['POST'])
def delete_medicine():
    medicine_id = request.form.get('medicine_id')

    conn = sqlite3.connect('medicine.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM medicines WHERE id = ?", (medicine_id,))
    conn.commit()
    conn.close()

    flash('Medicine deleted successfully!', 'success')  # Flash success message
    return redirect(url_for('admin_dashboard'))

@app.route('/logout', methods=['POST'])
def logout():
    # Clear the session or any relevant authentication data
    session.clear()  # Assuming you use Flask sessions for admin login
    return redirect(url_for('index'))  # Redirect to the home page (index.html)

# Define the assess route
@app.route('/assess')
def assess():
    return render_template('assess.html')

# Define the recommend route (POST method)
@app.route('/recommend', methods=['POST'])
def recommend():
    symptoms = request.form.get('symptoms', '')
    maintenance_meds = request.form.get('maintenance_medicine', '').split(',')
    allergies = request.form.get('allergy', '').split(',')
    
    recommendations = recommend_medicine(symptoms, maintenance_meds, allergies)
    return render_template('assess.html', recommendations=recommendations)



if __name__ == "__main__":
    app.run(debug=True)