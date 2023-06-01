import fuzzy
import json
from nltk import edit_distance
from SoundsLike.SoundsLike import Search


dmetaphone = fuzzy.DMetaphone()

with open('training_examples.json', encoding='utf-8') as json_file:
  training_examples = json.load(json_file)

for training_example in training_examples:
  sentence, label = training_example['sentence'], training_example['label']
  original_sentence = sentence[:]

  input_words = [word.strip().lower() for word in sentence.split(";")]
  input_metaphones = [dmetaphone(word)[0] for word in input_words]
  input_metaphones_secondary = [dmetaphone(word)[1] for word in input_words]

  
# input_words ['i', 'am', 'willing', 'to', 'enter']
# input_metaphones [b'A', b'AM', b'ALNK', b'T', b'ANTR']
# input_metaphones_secondary [None, None, b'FLNK', None, None]

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
      # secondary_metaphone = input_metaphones_secondary[i]
      # print('Secondary Metaphone', secondary_metaphone)
      try:
        secondary_words = Search.closeHomophones(input_words[i])
      except ValueError:
        secondary_words = []
      secondary_words = [word.lower() for word in secondary_words]
      secondary_words.sort(key=lambda x: edit_distance(x, label_words[i]))
      if input_words[i] in secondary_words:
        secondary_words.remove(input_words[i])
      if len(secondary_words) == 0:
        continue
      replacement_word = secondary_words[0]
      secondary_metaphone = dmetaphone(replacement_word)[0]
      # print(f'Word: {input_words[i]}, Secondary Words: {secondary_words}, Replacement Word: {replacement_word}')

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
  if fixed and phonetic_distance != 0 and phonetic_distance > old_phonetic_distance:
      sentence = ";".join(sentence_list)
      print("Original Transcription:", original_sentence)
      print("Corrected Transcription:", sentence)
      print("Label:", label)
      print("Old Phonetic Distance:", old_phonetic_distance)
      print("Phonetic Distance:", phonetic_distance, phonetic_distances)

      print()
