import sqlite3

DATABASE = 'search_history.db'

# Initialize SQLite database
def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                value TEXT NOT NULL
            )
        ''')

# Connect to the SQLite database
def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

# Insert search history entry into the database
def insert_search_entry(title, value):
    db = get_db()
    db.execute('INSERT INTO search_history (title, value) VALUES (?, ?)', (title, value))
    db.commit()
    db.close()

# Retrieve search history entries from the database
def get_search_history():
    db = get_db()
    search_history = db.execute('SELECT * FROM search_history').fetchall()
    db.close()
    return search_history

# Initialize the database when the app starts
init_db()
