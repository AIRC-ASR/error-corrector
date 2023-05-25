import fuzzy
import json
from nltk import edit_distance

dmetaphone = fuzzy.DMetaphone()

with open('training_examples.json', encoding='utf-8') as json_file:
  training_examples = json.load(json_file)

for training_example in training_examples:
  sentence, label = training_example['sentence'], training_example['label']

  input_words = [word.strip().lower() for word in sentence.split(";")]
  input_metaphones = [dmetaphone(word)[0] for word in input_words]
  input_metaphones_secondary = [dmetaphone(word)[1] for word in input_words]

  label_words = [word.strip().lower() for word in label.split(";")]
  label_metaphones = [dmetaphone(word)[0] for word in label_words]

  phonetic_distances = []
  for input_metaphone, label_metaphone in zip(input_metaphones, label_metaphones):
    if input_metaphone is None or label_metaphone is None:
      phonetic_distances.append(0)
    else:
      phonetic_distances.append(edit_distance(input_metaphone, label_metaphone))

  for i, phonetic_distance in enumerate(phonetic_distances):
    if phonetic_distance != 0:
      input_metaphones[i] = input_metaphones_secondary[i]

  phonetic_distances = []
  for input_metaphone, label_metaphone in zip(input_metaphones, label_metaphones):
    if input_metaphone is None or label_metaphone is None:
      phonetic_distances.append(0)
    else:
      phonetic_distances.append(edit_distance(input_metaphone, label_metaphone))

  phonetic_distance = sum(phonetic_distances) / len(phonetic_distances)
  if phonetic_distance != 0:
    print("Transcription:", sentence)
    print("Label:", label)
    print("Phonetic Distance:", phonetic_distance, phonetic_distances)
    print()
