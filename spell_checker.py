import json
import re
import nltk
from nltk.metrics.distance import edit_distance
# Import necessary libraries
from difflib import get_close_matches

import enchant

# Create an instance of the Enchant dictionary for English
en_dict = enchant.Dict("en_US")

# Function to generate alternative suggestions for a misspelled word
def generate_alternative_suggestions(word):
    suggestions = en_dict.suggest(word)
    return suggestions

def is_word_uncertain(word):
  return not en_dict.check(word)

# Function to score alternative suggestions
def score_suggestions(word, suggestions):
    # Calculate the similarity score between the word and suggestions
    scores = [(suggestion, similarity_score(word, suggestion)) for suggestion in suggestions]
    return scores

# Function to calculate the similarity score between two words
def similarity_score(word1, word2):
    # You can use different similarity metrics here (e.g., Levenshtein distance, Jaccard similarity)
    # For simplicity, let's use a basic approach based on the length of common characters
    common_chars = set(word1) & set(word2)
    score = len(common_chars) / max(len(word1), len(word2))
    return score

def improve_hypothesis(hypothesis):
    improved_words = []
    words = hypothesis.split()
    
    for word in words:
        if is_word_uncertain(word):
            suggestions = generate_alternative_suggestions(word)
            scored_suggestions = score_suggestions(word, suggestions)
            
            if scored_suggestions:
                best_suggestion, best_score = max(scored_suggestions, key=lambda x: x[1])
                if best_score > similarity_score(word, word):
                    improved_words.append(best_suggestion)
                else:
                    improved_words.append(word)  # Keep the original word if no better suggestions found
            else:
                improved_words.append(word)  # Keep the original word if no suggestions found
        else:
            improved_words.append(word)
    
    improved_hypothesis = " ".join(improved_words)

    return improved_hypothesis

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
    
    sentence = sentence.replace(';', '')
    sentence = re.sub(r"\s+", " ", sentence)

    label = label.replace(';', '')
    label = re.sub(r"\s+", " ", label)

    sentences.append(sentence)
    labels.append(label)

  total_wer = 0
  for sentence, label in zip(sentences, labels):
    wer = calculateWER(label, sentence)
    total_wer += wer
  
  overall_baseline_wer = total_wer / len(sentences)
  print(f'Overall Baseline WER: {overall_baseline_wer:.2f}%')

  total_wer = 0
  for sentence, label in zip(sentences, labels):
    hypothesis = improve_hypothesis(sentence)
    wer = calculateWER(label, hypothesis)
    total_wer += wer

  overall_improved_wer = total_wer / len(sentences)
  print(f'Overall Improved WER: {overall_improved_wer:.2f}%')