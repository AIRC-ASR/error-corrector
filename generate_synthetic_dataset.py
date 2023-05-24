
import glob
import random


def synthetic_step(sentence, error_prob):
    sentence_chars = list(sentence)
    num_chars = len(sentence_chars)
    error_types = ["insertion", "deletion", "substitution"]
    error_type = None
    error_position = None

    if random.random() < error_prob:
      error_type = random.choice(error_types)
      error_position = random.randint(0, num_chars - 1)

      if error_type == "insertion":
        char = random.choice(sentence_chars)
        sentence_chars.insert(error_position, char)
      elif error_type == "deletion":
        del sentence_chars[error_position]
      elif error_type == "substitution":
        char = random.choice(sentence_chars)
        sentence_chars[error_position] = char
    else:
      error_type = "no_change"

    return "".join(sentence_chars), sentence, error_type, error_position


def generate_synthetic_dataset(sentences, error_prob):
  synthetic_dataset = []
  for sentence in sentences:
    synthetic_sentence, sentence, error_type, error_position = synthetic_step(sentence, error_prob)
    synthetic_dataset.append((synthetic_sentence, sentence, error_type, error_position))

  return synthetic_dataset


def load_ground_truth_text(dataset_path):
  ground_truth_texts = []
  file_paths = glob.glob(dataset_path + "/**/*.txt", recursive=True)

  for file_path in file_paths:
    with open(file_path, "r") as file:
      lines = file.readlines()
      for line in lines:
        parts = line.strip().split(" ")
        ground_truth_text = " ".join(parts[1:]).lower()
        ground_truth_texts.append(ground_truth_text)

  return ground_truth_texts

# if __name__ == "__main__":
#   # Example usage
#   dataset_path = "LibriSpeech/"  # Replace with the path to your LibriSpeech dataset directory
#   ground_truth_texts = load_ground_truth_text(dataset_path)

#   # Print some example ground truth texts
#   print("Number of Training Examples:", len(ground_truth_texts))

#   error_prob = 0.5
#   synthetic_dataset = generate_synthetic_dataset(ground_truth_texts, error_prob)

#   # Print original and synthetic sentences
#   for i, (original, synthetic) in enumerate(zip(ground_truth_texts, synthetic_dataset)):
#     print(f"Original {i + 1}: {original}")
#     print(f"Synthetic {i + 1}: {synthetic}")
#     print()
  
