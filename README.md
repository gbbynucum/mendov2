# 💊 Medicine Recommendation & E-Pharmacy System  

## 📌 Overview  
This project is a **Flask-based web application** that recommends medicines based on user-entered symptoms, while also considering **maintenance medicines** and **allergies**.  

It combines:  
- **Natural Language Processing (NLP)** with **NLTK** and **TF-IDF + Cosine Similarity** to match symptoms to possible medications.  
- A small **e-commerce pharmacy system** with a shopping cart, payment simulation, and admin panel to manage medicines.  

Users can:  
- Enter symptoms and get recommended medicines with alternatives.  
- Shop medicines and manage their cart.  
- Pay (simulated payment) for items in the cart.  
- Admins can add, update, or delete medicines from the database.  

---

## ⚙️ Features  
✅ Symptom-based medicine recommendation  
✅ Considers maintenance medicines and allergies (avoid unsafe drugs)  
✅ Synonym expansion for local terms (English, Tagalog, Cebuano)  
✅ Medicine shop with stock, prices, images, effects, side effects  
✅ Shopping cart with add/remove/update  
✅ Payment simulation with fund validation  
✅ Admin panel for medicine management (CRUD)  
✅ SQLite database integration  

---

## 🛠️ Setup Instructions

### 1 Clone the repository  
```bash
git clone https://github.com/your-username/mendov2.git
cd mendov2

```

### 2 Clone the repository  
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3 Install dependencies

```bash
pip install -r requirements.txt
```
(Create requirements.txt with packages: Flask, pandas, nltk, scikit-learn, etc.)



### 4 Run the application

```bash
python app.py
```

Flask will start on:

```bash
http://127.0.0.1:5000/
```













