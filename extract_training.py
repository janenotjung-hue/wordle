import re
import pandas as pd

# Function to extract values from text using regular expressions
def extract_values(text):
    ent_loss_pattern = re.compile(r'\bent_loss\s*\|\s*([-+]?\d*(?:\.\d+)?(?:[eE][-+]?\d+)?)')
    entropy_pattern = re.compile(r'\bentropy\s*\|\s*([-+]?\d*(?:\.\d+)?(?:[eE][-+]?\d+)?)')
    loss_pattern = re.compile(r'\bloss\s*\|\s*([-+]?\d*(?:\.\d+)?(?:[eE][-+]?\d+)?)')
    
    ent_loss = ent_loss_pattern.search(text)
    entropy = entropy_pattern.search(text)
    loss = loss_pattern.search(text)
    
    return {
        'ent_loss': float(ent_loss.group(1)) if ent_loss else None,
        'entropy': float(entropy.group(1)) if entropy else None,
        'loss': float(loss.group(1)) if loss else None,
    }


# Read the text file
with open('training_for_1000_epochs.txt', 'r') as file:
    content = file.read()

# Split the content by group indicators (assuming each group starts with '0batch')
groups = content.split('0batch ')

# Extract values from each group
data = []
for group in groups:
    if group.strip():
        values = extract_values(group)
        data.append(values)

# Create a DataFrame
df = pd.DataFrame(data)
# Display the DataFrame
print(df)
# Save the DataFrame to a CSV file if needed
df.to_csv('extracted_data.csv', index=False)
