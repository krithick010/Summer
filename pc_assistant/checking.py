import sqlite3
c = sqlite3.connect("data/file_index.db")
q = lambda s: c.execute(s).fetchall()
print("total", q("select count(*) from files"))
print("onedrive", q("select count(*) from files where path like '%OneDrive%'"))
print("wallpapers", q("select path from files where name_norm like '%wallpaper%' limit 5"))
print("tree", q("select path from files where name_norm like 'tree%' limit 5"))