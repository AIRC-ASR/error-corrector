import json
import re
import nltk
import enchant
from nltk import edit_distance

# Create an instance of the Enchant dictionary for English
en_dict = enchant.Dict("en_US")

# Function to improve hypothesis 1
def improve_hypothesis(hypothesis):
    improved_words = []
    words = hypothesis.split()

    for word in words:
        if not en_dict.check(word):
            suggestions = en_dict.suggest(word)

            if suggestions:
                best_suggestion = suggestions[0]  # Select the first suggestion as the best
                improved_words.append(best_suggestion)
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