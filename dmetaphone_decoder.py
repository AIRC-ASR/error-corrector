import fuzzy
import json
from nltk import edit_distance

dmetaphone = fuzzy.DMetaphone()

with open('training_examples.json', encoding='utf-8') as json_file:
  training_examples = json.load(json_file)

for training_example in training_examples:
  sentence, label = training_example['sentence'], training_example['label']
  original_sentence = sentence[:]

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

  for word in label.split(";"):
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

  old_phonetic_distance = sum(phonetic_distances) / len(phonetic_distances)
  sentence_list = sentence.lower().strip().split(";")
  fixed = False
  for i, phonetic_distance in enumerate(phonetic_distances):
    if phonetic_distance != 0:
      secondary_metaphone = input_metaphones_secondary[i]
      # print('Secondary Metaphone', secondary_metaphone)
      if secondary_metaphone is not None:
        secondary_words = list(metaphone_dict[secondary_metaphone])
        secondary_words.sort(key=lambda x: edit_distance(x, label_words[i]))
        secondary_words.remove(sentence_list[i].strip())
        if len(secondary_words) == 0:
          continue
        replacement_word = secondary_words[0]
        print('Secondary Words', secondary_words)
        print('Replacement Word', replacement_word, sentence_list[i].strip(), replacement_word == sentence_list[i].strip())
        # print("Secondary Metaphone:", secondary_metaphone, secondary_words, replacement_word)
        sentence_list[i] = replacement_word.upper()

        input_metaphones[i] = secondary_metaphone
        fixed = True

  phonetic_distances = []
  for input_metaphone, label_metaphone in zip(input_metaphones, label_metaphones):
    if input_metaphone is None or label_metaphone is None:
      phonetic_distances.append(0)
    else:
      phonetic_distances.append(edit_distance(input_metaphone, label_metaphone))

  phonetic_distance = sum(phonetic_distances) / len(phonetic_distances)
  if fixed and old_phonetic_distance == phonetic_distance:
  # if phonetic_distance != 0:
    sentence = ";".join(sentence_list)
    print("Original Transcription:", original_sentence)
    print("Corrected Transcription:", sentence)
    print("Label:", label)
    print("Old Phonetic Distance:", old_phonetic_distance)
    print("Phonetic Distance:", phonetic_distance, phonetic_distances)

    print()
