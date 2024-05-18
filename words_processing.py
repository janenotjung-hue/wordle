import csv

# Function to read words from a CSV file
def read_words_from_csv(filename):
    words = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # Assuming each row contains one word
            words.append(row[0])
    return words

# Function to write words to a CSV file
def write_words_to_csv(words, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        for word in words:
            csvwriter.writerow([word])

# Read words from the two files
words1 = read_words_from_csv('data/valid_guesses.csv')
words2 = read_words_from_csv('data/valid_solutions.csv')

# Combine the lists of words
combined_words = words1 + words2

# Remove duplicates
combined_words = list(set(combined_words))

# Sort the words
combined_words.sort()

# Write the sorted words to a new CSV file
write_words_to_csv(combined_words, 'data/all_words.csv')
