import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

file_path = 'Data/CleanedDataForAnalysis.xlsx'
data = pd.read_excel(file_path)

data = data.rename(columns=lambda x: x.strip())

filtered_data = data.groupby('SubCategory').head().copy()

def classify_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

filtered_data['sentiment'] = filtered_data['summary'].apply(classify_sentiment)
sentiment_by_category = filtered_data.groupby(['year', 'Category', 'sentiment']).size().unstack(fill_value=0)

def box_counting_fractal_dimension(data):
    data = (data - np.min(data)) / (
                np.max(data) - np.min(data) + 1e-8)  # Adding a small value to avoid division by zero
    box_sizes = np.logspace(0.01, 1, num=10, base=2)
    counts = []

    for size in box_sizes:
        count = 0
        for i in range(0, len(data), int(size)):
            if np.any(data[i:i + int(size)]):
                count += 1
        counts.append(count)

    counts = np.array(counts)
    non_zero_indices = counts > 0
    box_sizes = box_sizes[non_zero_indices]
    counts = counts[non_zero_indices]

    if len(box_sizes) < 2:
        return np.nan
    return linregress(np.log(box_sizes), np.log(counts))[0]

def compute_fractal_dimensions_by_category(category_data):
    fractal_dims = {}
    for sentiment in category_data.columns:
        fractal_dims[sentiment] = box_counting_fractal_dimension(category_data[sentiment])
    return fractal_dims


categories = sentiment_by_category.index.get_level_values('Category').unique()

# Plotting Separate Graphs for Each Category
for category in categories:
    category_data = sentiment_by_category.xs(category, level='Category')
    category_fractal_dimensions = compute_fractal_dimensions_by_category(category_data)
    years = category_data.index

    plt.figure(figsize=(10, 6))

    for sentiment in category_data.columns:
        plt.plot(
            years,
            category_data[sentiment],
            label=f'{sentiment} (Fractal Dimension: {category_fractal_dimensions[sentiment]:.2f})'
        )

    plt.title(f'Sentiment Trends with Fractal Dimensions Over the Years for {category}')
    plt.xlabel('Year')
    plt.ylabel('Count of Sentiments')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
