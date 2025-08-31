import sqlite3

# Connect to your SQLite database
conn = sqlite3.connect('medicine.db')
cursor = conn.cursor()

# Execute the query to get the list of tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Print the list of tables
print("Tables in the database:", tables)

# Close the connection
conn.close()
