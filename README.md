

# Imitation Learning (IL) to Solve Wordle

OpenAI Gym for Wordle, plus Behavioural Cloning model applied for basic IL.



## Code Architecture

the folders checkpoints, results, training_information are for IL:
- checkpoints: stages of the model saved during training at diff epochs (20 cps).
- results: text output of each cp being tested on 100 random games.
- training_information: performance metrics recorded at each cp.

data folder:
- game_history: previous games of wordle used as expert demos (collect.py parses all the info to trajectories_all.npy).
- valid_guesses: words that are valid guesses but not solutions.
- valid_solutions: words that can be guesses AND solutions.
- all_words: combination of both lists (used words_processing.py to combine)

General:
- Wordle.py: code to create gymnasium environment.
- util.py: util functions to create sessions for player and model for one game of wordle. 
- main.py: makes a session to let player/model compete with each other.
- test.py: testing model on 10 random games after being trained.
## Acknowledgements

 - [gym-wordle](https://pypi.org/project/gym-wordle/)
 - [Wordle](https://github.com/preritdas/wordle)
 - [Wordle Solving with Deep Reinforcement Learning](https://andrewkho.github.io/wordle-solver/)
 - [Wo\[R\]d\[L\]e](https://github.com/harsh788/woRdLe)

