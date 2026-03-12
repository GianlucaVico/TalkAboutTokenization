#!/usr/bin/env python3
# Classify the papers in the diff file
import paper_dataset
import re
import pandas as pd
import llm_filter
import tqdm
tqdm.tqdm.pandas()
# # Load 
# df = paper_dataset.load("data/papers_diff.jsonl")

# # RE
# print("RE filtering...")
# recent = df[df['year'] >= 2016]
# keywords = re.compile(r"token|sub-?word|segment")
# tokenization = recent[recent["abstract"].apply(lambda x: x is not None and re.search(keywords, x.lower()) is not None)]

# print(f"Number of papers in the dataset: {len(df)}")
# print(f"Number of papers since 2016: {len(recent)}")
# print(f"Number of papers on tokenization since 2016: {len(tokenization)}")

# # Filter
# print("LLM filtering...")
# abstracts = tokenization['abstract'].tolist()
# titles = tokenization['title'].tolist()
# tokenization['topic'] = list(llm_filter.filter_topic(titles, abstracts))
# df = df.merge(tokenization[['bibtex_id', 'topic']], on='bibtex_id', how='left')

# # Save intermediate the results
# df.to_json('data/papers_diff_topic.jsonl', orient='records', lines=True)


# Classify
print("Classifying papers...")
df = paper_dataset.load("data/papers_diff_topic.jsonl")
real_tokenization = df[df['topic'] == True]
real_tokenization = real_tokenization.copy()

real_tokenization['tokenization'] = real_tokenization.progress_apply(lambda x: llm_filter.is_tokeniser(x['title'], x['abstract']), axis=1)
real_tokenization['evaluation'] = real_tokenization.progress_apply(lambda x: llm_filter.is_evaluation(x['title'], x['abstract']), axis=1)
real_tokenization['survey'] = real_tokenization.progress_apply(lambda x: llm_filter.is_survey(x['title'], x['abstract']), axis=1)
real_tokenization['application'] = ~(real_tokenization['tokenization'] | real_tokenization['evaluation'] | real_tokenization['survey'])
# Save the results
real_tokenization.to_json('data/real_tokenization_diff.jsonl', orient='records', lines=True)
