import sqlite3

my_connection = sqlite3.Connection("zdeleteme/streamline_db.db")

with my_connection:
    my_connection.execute('DELETE FROM ALRound')