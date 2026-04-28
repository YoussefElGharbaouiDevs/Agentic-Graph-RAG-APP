from neo4j import GraphDatabase

uri      = "neo4j+ssc://b8e0d8d2.databases.neo4j.io"
user     = "b8e0d8d2"
password = "61sqP3d6ZSM8kh56Ppt-WSPGNlbe16gYrGXoV-Flp7I"

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print("SUCCESS!")
    driver.close()
except Exception as e:
    print("ERROR:", e)