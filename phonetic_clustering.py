import re
import json
from nltk import edit_distance
import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

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
  X_original = df.loc[:, ['sentence', 'label']].values

  vectorizer = TfidfVectorizer(stop_words='english')
  X = vectorizer.fit_transform(sentences)

  wcss = []
  for i in range(1,11): 
    kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )

    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

  plt.plot(range(1,11),wcss)
  plt.title('The Elbow Method Graph')
  plt.xlabel('Number of clusters')
  plt.ylabel('WCSS')
  plt.savefig('elbow_method.png')

  kmeans = KMeans(n_clusters=7, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
  y_kmeans = kmeans.fit_predict(X)

  # plt.scatter(X_original[y_kmeans==0, 0], X_original[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
  # plt.scatter(X_original[y_kmeans==1, 0], X_original[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
  # plt.scatter(X_original[y_kmeans==2, 0], X_original[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
  # plt.scatter(X_original[y_kmeans==3, 0], X_original[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
  # plt.scatter(X_original[y_kmeans==4, 0], X_original[y_kmeans==4, 1], s=100, c='magenta', label ='Cluster 5')
  # plt.scatter(X_original[y_kmeans==5, 0], X_original[y_kmeans==5, 1], s=100, c='darkorange', label ='Cluster 6')
  # plt.scatter(X_original[y_kmeans==6, 0], X_original[y_kmeans==6, 1], s=100, c='darkviolet', label ='Cluster 7')

  # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
  # plt.title('Clusters of Customers')
  # plt.xlabel('Annual Income(k$)')
  # plt.ylabel('Spending Score(1-100')
  # plt.savefig('clusters.png')

  order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
  terms = vectorizer.get_feature_names_out()
  for i in range(7):
      print("Cluster %d:" % i),
      for ind in order_centroids[i, :10]:
          print(' %s' % terms[ind]),
      print