import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


prompts = pd.read_csv("movielens_benchmark_prompts.csv")
print(prompts.head())
print(prompts.columns)


prompts['length'] = prompts['vocab'].str.len()

print("\n===== PROMPTS — INFO =====")
print(prompts.info())

print("\n===== PROMPTS — FIRST 5 ROWS =====")
print(prompts.head())

print("\n===== PROMPTS — MISSING VALUES PER COLUMN =====")
print(prompts.isna().sum())

print("\n===== PROMPTS — NUMBER OF UNIQUE PROMPTS =====")
print(prompts['vocab'].nunique())

print("\n===== PROMPTS — TOP 20 MOST FREQUENT PROMPTS =====")
print(prompts['vocab'].value_counts().head(20))

print("\n===== PROMPTS — DESCRIPTION LENGTH STATISTICS =====")
print(prompts['length'].describe())



min_prompt_len = prompts['length'].min()
shortest_prompts = prompts[prompts['length'] == min_prompt_len]

print("\n===== SHORTEST PROMPT(S) =====")
print(f"Length: {min_prompt_len}")
print(shortest_prompts['vocab'])

max_prompt_len = prompts['length'].max()
longest_prompts = prompts[prompts['length'] == max_prompt_len]

print("\n===== LONGEST PROMPT(S) =====")
print(f"Length: {max_prompt_len}")
print(longest_prompts['vocab'])

