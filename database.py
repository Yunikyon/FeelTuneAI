import sqlite3

conn = sqlite3.connect('feeltune.db')
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS users")
cursor.execute("DROP TABLE IF EXISTS musics")
cursor.execute("DROP TABLE IF EXISTS musics_listened")
cursor.execute("DROP TABLE IF EXISTS user_musics")

# Create table
cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    progress INTEGER NOT NULL)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS musics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    valence REAL NOT NULL,
    arousal REAL NOT NULL)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS musics_listened (
    user_id INTEGER NOT NULL,
    music_id INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (music_id) REFERENCES musics(id))''')

cursor.execute('''CREATE TABLE IF NOT EXISTS user_musics (
    user_id INTEGER NOT NULL,
    music_id INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (music_id) REFERENCES musics(id))''')


conn.commit()
conn.close()


