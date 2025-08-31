import sqlite3

# Connect to the database
conn = sqlite3.connect('medicine.db')
cursor = conn.cursor()

# Create the admins table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS admins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )
''')

# Insert a default admin user (optional)
cursor.execute('''
    INSERT OR IGNORE INTO admins (username, password) VALUES (?, ?)
''', ('admin', 'adminpass'))  # Replace with your desired default credentials

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Admins table created successfully, and default admin added if not already present.")
