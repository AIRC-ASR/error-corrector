import re
import json
from nltk import edit_distance
import matplotlib.pyplot as plt
import pandas as pd
import csv
import Levenshtein
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

def cluster_strings(strings):
    # Preprocess strings
    preprocessed_strings = strings
    
    # Calculate pairwise distance matrix
    distance_matrix = pdist(preprocessed_strings, metric='levenshtein')
    
    # Perform hierarchical agglomerative clustering
    linkage_matrix = linkage(distance_matrix, method='complete')
    
    # Determine clusters based on a distance threshold
    threshold = 0.5  # Adjust this threshold based on your data
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')
    
    return clusters

def calculateWER(reference, hypothesis):
  reference_words = reference.split()
  hypothesis_words = hypothesis.split()
  num_reference_words = len(reference_words)

  # Calculate the edit distance between the reference and hypothesis sentences
  edit_dist = edit_distance(reference_words, hypothesis_words)

  # Calculate the WER as the edit distance divided by the number of reference words
  wer = edit_dist / num_reference_words * 100

  return wer

if __name__ == '__main__':
  with open('training_examples.json', encoding='utf-8') as json_file:
    training_examples = json.load(json_file)

  sentences = []
  labels = []
  for training_example in training_examples:
    sentence, label = training_example['sentence'], training_example['label']
    
    sentence = sentence.replace(';', '').strip().lower()
    sentence = re.sub(r"\s+", " ", sentence)

    label = label.replace(';', '').strip().lower()
    label = re.sub(r"\s+", " ", label)

    sentences.append(sentence)
    labels.append(label)

  with open('training_examples.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['sentence', 'label'])
    for sentence, label in zip(sentences, labels):
      writer.writerow([sentence, label])

  total_wer = 0
  for sentence, label in zip(sentences, labels):
    wer = calculateWER(label, sentence)
    total_wer += wer
  
  overall_baseline_wer = total_wer / len(sentences)
  print(f'Overall Baseline WER: {overall_baseline_wer:.2f}%')

  df = pd.read_csv('training_examples.csv')
  print(df.info())
  print(df.head())

  # Select the sentence and label columns only
  X = df.loc[:, ['sentence', 'label']].values

  cluster_strings(X)