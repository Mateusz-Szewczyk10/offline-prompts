import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

queries = pd.read_csv("movielens_query.csv")
print(queries.head())
print(queries.columns)

queries['length'] = queries['query'].str.len()

print("\n===== QUERIES — INFO =====")
print(queries.info())

print("\n===== QUERIES — FIRST 5 ROWS =====")
print(queries.head())

print("\n===== QUERIES — MISSING VALUES PER COLUMN =====")
print(queries.isna().sum())

print("\n===== QUERIES — NUMBER OF UNIQUE QUERIES =====")
print(queries['query'].nunique())

print("\n===== QUERIES — TOP 20 MOST FREQUENT QUERIES =====")
print(queries['query'].value_counts().head(20))

print("\n===== QUERIES — QUERY LENGTH STATISTICS =====")
print(queries['length'].describe())


min_query_len = queries['length'].min()
shortest_queries = queries[queries['length'] == min_query_len]

print("\n===== SHORTEST QUERY(IES) =====")
print(f"Length: {min_query_len}")
print(shortest_queries['query'])

# Longest query(ies)
max_query_len = queries['length'].max()
longest_queries = queries[queries['length'] == max_query_len]

print("\n===== LONGEST QUERY(IES) =====")
print(f"Length: {max_query_len}")
print(longest_queries['query'])