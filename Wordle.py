import random
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Tuple, MultiDiscrete, Box
from typing import Optional
#TODO CHANGE SPACES.DISCRETE TO DISCRETE

class WordleEnv(gym.Env):
    
    metadata = {
        "render_modes": ["human", ],
    }

    def __init__(self, word_length=5, max_attempts=6, subset_size=None):
        self.word_length = word_length
        self.max_attempts = max_attempts
        self.target_word = ''
        self.attempts_left = 0
        self.attempts = 0
        self.current_guess = ''

        # Opening the txt file containing possible words and get a random subset of them if necessary
        with open('data/valid_solutions.csv', 'r') as f:
        #with open('data/wordle_subset.txt', 'r') as f:
            words = [word.strip().upper() for word in f.readlines() if len(word.strip()) == word_length]
            if subset_size is not None:
                words = self.get_random_subset(words, subset_size)
            self.words = words

        with open('data/valid_guesses.csv', 'r') as f:
        #with open('data/wordle_subset.txt', 'r') as f:
            guessable_words = [word.strip().upper() for word in f.readlines() if len(word.strip()) == word_length]
            self.guessable_words = guessable_words

        # State space has 78 dimensions (3 for each letter, gray, yellow, and green states)
        self.state_size = 78
        self.observation_space = MultiDiscrete([2]*78)
        # Possible actions are the number of words in the dataset
        self.action_size = len(self.guessable_words)
        self.action_space = Discrete(self.action_size)
        # Current state starts as all zeros one hot encoded matrix, then it will be built after each move
        self.state = np.zeros(self.state_size, dtype=np.int32)
        
    # This function removes incompatible words based on current guesses.
    def remove_incompatible_words(self, current_guess):
        new_available_actions = []
        for i in self.available_actions:
            word = self.guessable_words[i]
            compatible = True
            for idx, (guess_char, target_char) in enumerate(zip(current_guess, self.target_word)):
                if guess_char == target_char and word[idx] != guess_char:
                    compatible = False
                    break
                elif guess_char != target_char and word[idx] == guess_char:
                    compatible = False
                    break
            if compatible:
                new_available_actions.append(i)

        # Ensure at least one word is left in the available word list
        if len(new_available_actions) > 0:
            self.available_actions = new_available_actions

    # This function masks action for the incompatible actions.
    def mask_action(self, action):
        if action in self.available_actions:
            self.available_actions.remove(action)
    
    # This function gets a random subset of words
    def get_random_subset(self, words, subset_size):
        return random.sample(words, subset_size)
    
    # This function chooses a random number between 0 and length of dataset, which will be transformed into word based on the index.
    def get_random_action(self):
        return random.randint(0, self.action_size - 1)

    # Before starting each episode, the environment is resetted to give the initial conditions.
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.target_word = random.choice(self.words)
        print(type(self.target_word))
        self.attempts_left = self.max_attempts
        self.attempts = 0
        self.current_guess = '_' * self.word_length
        self.available_actions = list(range(self.action_size))
        #print(len(self.available_actions))
        self.state = np.zeros(self.state_size, dtype=np.int32)
        #TODO: check if it should return self.state
        return self.get_state(), {}

    def set_target_word(self, seed):
        self.target_word = self.words[seed]

    # Each time we make an action (make a guess), we check how many of the letters are correct.
    def step(self, action):
        self.current_guess = self.guessable_words[action]
        self.mask_action(action)  # Mask the taken action
        self.attempts += 1
        reward = 0
        done = False
        # If the guess is correct, +10 reward.
        if self.current_guess == self.target_word:
            reward = 10.0
            done = True
        # If some of the letters are correct, give intermediate reward for the number of correct letters [1,4]
        else:
            correct_letters = sum([1 for guessed_letter, target_letter in zip(self.current_guess, self.target_word) if guessed_letter == target_letter])
            reward = 1.0 * correct_letters
            
            self.attempts_left -= 1
            # If there is no attempts left, unsuccessful, -10 reward.
            if self.attempts_left <= 0:
                reward = -10.0
                done = True
        
        self.remove_incompatible_words(self.current_guess)
        observation = self.get_state()
        return observation, reward, done, done, {}
    
    # In each turn, get the new state based on the correctness of the letters
    def get_state(self):
        state = self.state
        # Check each letter of the guess
        for idx, letter in enumerate(self.current_guess):
            if letter == '_':
                break
            # If correct location and letter (green), that is allocated for 0,25
            if letter == self.target_word[idx]:
                state[(ord(letter) - 65)] = 1
            # If only correct letter (yellow), allocated for second 26 indices.
            elif letter in self.target_word:
                state[(ord(letter) - 65) + 26] = 1
            # If the letter is not in the word, allocated for the last 26 indices.
            else:
                state[(ord(letter) - 65) + 26*2] = 1
        return state

    def get_observation(self):
        state = self.state
        split_arrays = np.split(state, 3)
        sum_array = np.sum(split_arrays, axis=0)
        # Check each letter of the guess
        for idx, letter in enumerate(self.current_guess):
            if letter == '_':
                break
            # If correct location and letter (green), that is allocated for 0,25
            if letter == self.target_word[idx]:
                sum_array[(ord(letter) - 65)] = 2
            # If only correct letter (yellow), allocated for second 26 indices.
            elif letter in self.target_word:
                sum_array[(ord(letter) - 65)] = 1
            # If the letter is not in the word, allocated for the last 26 indices.
            else:
                sum_array[(ord(letter) - 65)] = 0
        return sum_array
        
    # Printing output purposes.
    def render(self):
        print(f"Current guess: {self.current_guess}")
        print(f"Target word: {self.target_word}")
        print(f"Attempts left: {self.attempts_left}")
