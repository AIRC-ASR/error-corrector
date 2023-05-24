import random
import torch.optim as optim
import string
from nltk.metrics.distance import edit_distance
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import numpy as np
import torch
import torch.nn as nn

class ASRErrorCorrectionEnvironment:
    def __init__(self, clean_sentence, error_prob):
        self.clean_sentence = clean_sentence
        self.current_sentence, self.error_position, self.new_char = self.generate_synthetic_errors(clean_sentence, error_prob)
        self.action_space = self.generate_action_space(clean_sentence)
        self.error_prob = error_prob
        self.done = False

    def generate_action_space(self, sentence):
        action_space = []

        # Define the types of actions
        action_types = ["insertion", "deletion", "substitution", "no_change"]

        # Generate all possible combinations of actions
        for action_type in action_types:
            if action_type == "no_change":
                action = {
                    "type": action_type,
                    "position": None,
                    "new_char": None,
                }
                action_space.append(action)
            elif action_type == "deletion":
                for position in range(len(sentence)):
                    action = {
                        "type": action_type,
                        "position": position,
                        "new_char": None,
                    }
                    action_space.append(action)
            elif action_type == "insertion":
                for position in range(len(sentence) + 1):
                    for char in string.ascii_lowercase:
                        action = {
                            "type": action_type,
                            "position": position,
                            "new_char": char,
                        }
                        action_space.append(action)
            elif action_type == "substitution":
                for position in range(len(sentence)):
                    for char in string.ascii_lowercase:
                        action = {
                            "type": action_type,
                            "position": position,
                            "new_char": char,
                        }
                        action_space.append(action)

        print("ACTIONS", len(action_space))
        return action_space



    def reset(self):
        self.current_sentence, self.error_position, self.new_char = self.generate_synthetic_errors(self.current_sentence, self.error_prob)
        self.done = False
        return self.current_sentence, self.error_position, self.new_char

    def generate_synthetic_errors(self, sentence, error_prob):
          sentence_chars = list(sentence)
          num_chars = len(sentence_chars)
          error_types = ["insertion", "deletion", "substitution"]
          error_type = None
          error_position = None
          new_char = None

          if random.random() < error_prob:
            error_type = random.choice(error_types)
            error_position = random.randint(0, num_chars - 1)

            if error_type == "insertion":
              char = random.choice(sentence_chars)
              sentence_chars.insert(error_position, char)
              new_char = char
            elif error_type == "deletion":
              del sentence_chars[error_position]
            elif error_type == "substitution":
              char = random.choice(sentence_chars)
              sentence_chars[error_position] = char
              new_char = char
          else:
            error_type = "no_change"

          return "".join(sentence_chars), error_position, new_char

    def step(self, action):
        # Unpack the action components
        action_type, action_position, action_new_char = action['type'], action['position'], action['new_char']

        # Apply the chosen action to the current sentence
        # Update the error position based on the action

        # Calculate the edit distance between the modified sentence and the clean sentence
        previous_distance = edit_distance(self.current_sentence, self.clean_sentence)

        # Update the current sentence based on the action
        if action_type == "insertion":
            # Apply insertion action
            self.current_sentence = self.current_sentence[:action_position] + action_new_char + self.current_sentence[action_position:]
        elif action_type == "deletion":
            # Apply deletion action
            self.current_sentence = self.current_sentence[:action_position] + self.current_sentence[action_position + 1:]
        elif action_type == "substitution":
            # Apply substitution action
            self.current_sentence = self.current_sentence[:action_position] + action_new_char + self.current_sentence[action_position + 1:]

        # Calculate the new edit distance
        current_distance = edit_distance(self.current_sentence, self.clean_sentence)

        # Calculate the reward
        reward = previous_distance - current_distance

        # Check if the episode is done
        self.done = (current_distance == 0)

        # Return the updated state, reward, and done flag
        return self.current_sentence, reward, self.done


    def get_state(self):
        # Return the current state
        return self.current_sentence, self.error_position, self.new_char

# Epsilon-greedy exploration strategy
def epsilon_greedy_action(q_values, epsilon, num_chars, action_space):
    # Generate random value for exploration
    if random.random() < epsilon:
        # Randomly select an action
        action_type = random.choice(["insertion", "deletion", "substitution", "no_change"])
        action_new_char = random.choice(string.ascii_lowercase)  # Choose a random lowercase character
        action_position = random.randint(0, num_chars - 1)  # Choose a random position

        if action_type == "no_change":
            action_new_char = None
            action_position = None
        elif action_type == "deletion":
            action_new_char = None

        return action_type, action_position, action_new_char
    else:
        # Select the action with the highest Q-value
        valid_indices = range(len(action_space))
        # print("SS", )
        max_q_value_index = np.argmax(q_values[valid_indices])
        print("MAX Q VALUE INDEX", max_q_value_index, q_values.shape)
        action_index = max_q_value_index
        action_type, action_position, action_new_char = action_space[action_index]


        return action_type, action_new_char, action_position



# Define the epsilon value for exploration
epsilon = 0
discount_factor = 0.9

# Define the learning rate and other optimizer parameters
learning_rate = 0.001
weight_decay = 0.0001



num_episodes = 1000
clean_sentence = "the quick brown fox jumps over the lazy dog"
error_prob = 0.5

# Initialize the T5 model and tokenizer
# Define a simple feed-forward neural network model
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

env = ASRErrorCorrectionEnvironment(clean_sentence, error_prob)
num_actions = len(env.action_space)
model = MLP(input_size=512, output_size=num_actions)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
tokenizer = T5Tokenizer.from_pretrained('t5-base')
loss_fn = nn.MSELoss()

# Training loop
for episode in range(num_episodes):
    # Reset the environment
    state = env.reset()
    done = False

    while not done:
        # Get the current state
        
        current_sentence, error_position, new_char = state
        print("AAA", current_sentence, error_position, new_char)

        # Tokenize the current sentence and convert it to a tensor
        synth_error_inputs = tokenizer.encode_plus(current_sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        input_ids = synth_error_inputs["input_ids"].to(torch.float32)

        # Forward pass through the model
        q_values = model(input_ids)

        # Choose the action with the highest Q-value
        action_index = torch.argmax(q_values, dim=1).item()
        action_type = env.action_space[action_index]
        print("ACTION INDEX", action_index, action_type)

        # Perform the action in the environment
        next_sentence, reward, done = env.step(action_type)

        # Tokenize the next sentence and convert it to a tensor
        next_inputs = tokenizer.encode_plus(next_sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        next_input_ids = next_inputs["input_ids"].to(torch.float32)

        # Forward pass through the model to get the Q-values of the next state
        next_q_values = model(next_input_ids)

        # Compute the target Q-value
        max_q_value = torch.max(next_q_values, dim=1)[0]
        target_q_value = reward + discount_factor * max_q_value

        # Compute the loss
        loss = loss_fn(q_values, target_q_value.unsqueeze(1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the state for the next iteration
        state = (next_sentence, env.error_position, env.new_char)

    # Print the episode number and final Q-values for monitoring
    print(f"Episode: {episode + 1}, Final Q-values: {q_values}")
