# Import libraries
from gector.gec_model import GecBERTModel
import json

# Create an instance of the model
model = GecBERTModel(vocab_path = "./data/output_vocabulary", model_paths = ["./gector/roberta_1_gectorv2.th"])

with open('training_examples.json', encoding='utf-8') as json_file:
  training_examples = json.load(json_file)

for training_example in training_examples:
  sentence, label = training_example['sentence'], training_example['label']
  batch = []
  sentence = sentence.lower().strip()
  label = label.lower().strip()
  batch.append(sentence.split())
  final_batch, total_updates = model.handle_batch(batch)
  updated_sentence = " ".join(final_batch[0])
  print(f"Original Sentence: {sentence}")
  print(f"Updated Sentence: {updated_sentence}")
  print(f"Label: {label}")
  print("")