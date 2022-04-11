import sqlite3 
import numpy

database = numpy.genfromtxt('UCS-Satellite-Database-1-1-2022.txt',dtype=None,delimiter='\t',usecols=range(0,26),skip_header=0,names=True,encoding='ISO-8859-15',invalid_raise=False)

link = sqlite3.connect('Satellites.db')
cursor = link.cursor()

#create DB here
cursor.execute('''CREATE TABLE orbital_elements
                    (NORADID int CHECK (NORADID > 0), orbitclass text, orbittype text, GEOlongitude decimal, perigee decimal, apogee decimal, eccentricity decimal, inclination decimal, period decimal, PRIMARY KEY(NORADID))''')

cursor.execute('''CREATE TABLE naming
                    (NORADID int CHECK (NORADID > 0), name text, country text, operator text, user text, purpose text, contractor text, PRIMARY KEY(NORADID))''')

cursor.execute('''CREATE TABLE technical_data
                    (NORADID int CHECK (NORADID > 0), launch_mass decimal, dry_mass decimal, power decimal, launch_date date, lifetime decimal, launch_site text, LV text, PRIMARY KEY(NORADID))''')

for i in range(0,database.shape[0]):
    noradid = str(database[i][25])
    name = str(database[i][0]).replace('\'', '^')
    country = str(database[i][2])
    operator = str(database[i][3]).replace('\'', '^')
    user = str(database[i][4]).replace('\'', '^')
    purpose = str(database[i][5]).replace('\'', '^')
    contractor = str(database[i][20]).replace('\'', '^')
    launchsite = str(database[i][22]).replace('\'', '^')
    lv = str(database[i][23]).replace('\'', '^')

    namevals = "'" + noradid + "','" + name + "','" + country + "','" + operator + "','" + user + "','" + purpose + "','" + contractor + "'"
    namestr = "INSERT OR IGNORE INTO naming VALUES (" + namevals + ")"
    orbitvals = "'" + noradid + "','" + str(database[i][7]) + "','" + str(database[i][8]) + "','" + str(database[i][9]) + "','" + str(database[i][10]) + "','" + str(database[i][11]) + "','" + str(database[i][12]) + "','" + str(database[i][13]) + "','" + str(database[i][14]) + "'"
    orbitstr = "INSERT OR IGNORE INTO orbital_elements VALUES (" + orbitvals + ")"
    techvals =  "'" + noradid + "','" + str(database[i][15]).replace(',', '') + "','" + str(database[i][16]).split(" ")[0].replace(',', '') + "','" + str(database[i][17]).replace(',', '') + "','" + str(database[i][18]) + "','" + str(database[i][19]) + "','" + launchsite + "','" + lv + "'"
    techstr = "INSERT OR IGNORE INTO technical_data VALUES (" + techvals + ")"

    cursor.execute(namestr)
    cursor.execute(orbitstr)
    cursor.execute(techstr)

link.commit()    
link.close()