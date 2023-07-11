import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('feeltune.db')
cursor = conn.cursor()

# Create the table if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS musics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    valence REAL NOT NULL,
    arousal REAL NOT NULL)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS user_musics (
    user_id INTEGER NOT NULL,
    music_id INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (music_id) REFERENCES musics(id))''')

# Read and insert data from the CSV file
delimiter = '~~~'
with open('./applications_musics_va.csv', 'r', encoding="utf-8") as file_obj:
    try:
        musics_df = pd.read_csv(file_obj, sep=delimiter, engine='python')
    except pd.errors.ParserError:
        musics_df = pd.read_csv(file_obj.replace(delimiter, ','), sep=',')

default_user_id = 0
for _, row in musics_df.iterrows():
    music_name = row['music_name']
    valence = row['music_valence']
    arousal = row['music_arousal']

    cursor.execute('''
        INSERT INTO musics (name, valence, arousal)
        VALUES (?, ?, ?)
    ''', (music_name, valence, arousal))
    music_id = cursor.lastrowid

    cursor.execute('''
        INSERT INTO user_musics (user_id, music_id)
        VALUES (?, ?)
        ''', (default_user_id, music_id))

# Commit the changes and close the connection
conn.commit()
conn.close()
