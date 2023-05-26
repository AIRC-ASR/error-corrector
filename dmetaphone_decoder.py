import fuzzy
import json
from nltk import edit_distance

dmetaphone = fuzzy.DMetaphone()

with open('training_examples.json', encoding='utf-8') as json_file:
  training_examples = json.load(json_file)

for training_example in training_examples:
  sentence, label = training_example['sentence'], training_example['label']

  # Create the metaphone dictionary mapping between phonetic codes and words
  metaphone_dict = {}
  for word in sentence.split(";"):
    word = word.strip().lower()
    metaphone = dmetaphone(word)[1]
    if metaphone is None:
      continue
    if metaphone not in metaphone_dict:
      metaphone_dict[metaphone] = set()
    metaphone_dict[metaphone].add(word)

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

  sentence_list = sentence.split(";")
  for i, phonetic_distance in enumerate(phonetic_distances):
    if phonetic_distance != 0:
      secondary_metaphone = input_metaphones_secondary[i]
      if secondary_metaphone is not None:
        secondary_words = list(metaphone_dict[secondary_metaphone])
        secondary_words.sort(key=lambda x: edit_distance(x, label_words[i]))
        replacement_word = secondary_words[0]
        # print("Secondary Metaphone:", secondary_metaphone, secondary_words, replacement_word)
        sentence_list[i] = replacement_word

      input_metaphones[i] = secondary_metaphone

  phonetic_distances = []
  for input_metaphone, label_metaphone in zip(input_metaphones, label_metaphones):
    if input_metaphone is None or label_metaphone is None:
      phonetic_distances.append(0)
    else:
      phonetic_distances.append(edit_distance(input_metaphone, label_metaphone))

  phonetic_distance = sum(phonetic_distances) / len(phonetic_distances)
  if phonetic_distance != 0:
    sentence = ";".join(sentence_list)
    print("Transcription:", sentence)
    print("Label:", label)
    print("Phonetic Distance:", phonetic_distance, phonetic_distances)
    print()
