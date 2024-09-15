import pandas as pd
import json

# Load JSON data from a file
with open('Data/arxivData.json') as file:
    data = json.load(file)

# Convert the JSON data to a DataFrame
df = pd.DataFrame(data)

# Remove the 'link' column if it exists
df.drop(columns=['link'], inplace=True, errors='ignore')

# Define a list of prefixes to filter relevant terms
allowed_prefixes = ['stat.', 'cs.', 'math.', 'astro-ph.', 'physics.', 'nlin.', 'q-bio.', 'cond-mat.', 'q-fin.']

# Function to filter terms from the 'tag' column based on allowed prefixes
def filter_tags(tag_str):
    terms = [term['term'] for term in eval(tag_str)]
    filtered_terms = [term for term in terms if any(term.startswith(prefix) for prefix in allowed_prefixes)]
    return ', '.join(filtered_terms)

# Modifying the 'author' column to show only author names separated by commas
df['author'] = df['author'].apply(lambda x: ', '.join([author['name'] for author in eval(x)]))

# Applying the filter function to the 'tag' column and create a 'SubCategory' column
df['SubCategory'] = df['tag'].apply(filter_tags)

# Creating a 'Category' column by extracting the part before the dot in each SubCategory
df['Category'] = df['SubCategory'].apply(lambda x: ', '.join(set([term.split('.')[0] for term in x.split(', ')])))

# Bifurcation process: separating subcategories based on categories
# Split the 'Category' and 'SubCategory' columns into lists
df['Category'] = df['Category'].apply(lambda x: [cat.strip() for cat in x.split(',')])
df['SubCategory'] = df['SubCategory'].apply(lambda x: [sub.strip() for sub in x.split(',')])

# Initialize a new dataframe to store the bifurcated data
bifurcated_data_full_with_all_columns = []

# Loop through each row in the entire dataset to bifurcate the subcategories based on the categories
df.drop(columns=['tag'], inplace=True, errors='ignore')

for _, row in df.iterrows():
    for category in row['Category']:
        # Filter subcategories that start with the category prefix
        subcategories = [sub for sub in row['SubCategory'] if sub.startswith(category)]
        for subcategory in subcategories:
            # Create a dictionary that includes all the original columns from the row, but with individual subcategories in new rows
            row_data = row.to_dict()
            row_data['Category'] = category
            row_data['SubCategory'] = subcategory
            bifurcated_data_full_with_all_columns.append(row_data)


# Convert the bifurcated data into a new DataFrame with all columns included
bifurcated_df_full_with_all_columns = pd.DataFrame(bifurcated_data_full_with_all_columns)

# Saving the bifurcated data into Excel for future analysis
file_path = "Data/CleanedDataForAnalysis.xlsx"
bifurcated_df_full_with_all_columns.to_excel(file_path, index=False)
