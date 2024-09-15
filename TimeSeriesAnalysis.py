import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_path = 'Data/CleanedDataForAnalysis.xlsx'
data = pd.read_excel(file_path)

yearly_publications = data.groupby('year').size()

# Plotting the time series data (yearly publication counts)
plt.figure(figsize=(10, 6))
plt.plot(yearly_publications.index, yearly_publications.values, marker='o', color='b')
plt.title('Number of Publications Over Time (Yearly)')
plt.xlabel('Year')
plt.ylabel('Number of Publications')
plt.grid(True)
plt.show()


# Fractal dimension calculation using the Box-Counting method with adjustment for zero counts
def adjusted_box_counting_fractal_dimension(data):
    N = len(data)
    L = np.logspace(0.01, np.log10(N / 2), num=30, endpoint=False, base=10).astype(int)
    counts = []

    for l in L:
        count = np.sum([np.ptp(data[i:i + l]) > 0 for i in range(0, N, l)])
        counts.append(max(count, 1e-5))
    coeffs = np.polyfit(np.log(L), np.log(counts), 1)
    fractal_dimension = -coeffs[0]

    return fractal_dimension


# Apply the fractal dimension calculation on the yearly publication counts
fd_yearly_adjusted = adjusted_box_counting_fractal_dimension(yearly_publications.values)
print("Fractal Dimension on the yearly publication : " + fd_yearly_adjusted)

# Group the data by year and category to count the number of publications for each category per year
category_publications = data.groupby(['year', 'Category']).size().unstack(fill_value=0)

# Plot the time series data for each category (yearly publication counts per category)
plt.figure(figsize=(12, 8))
for category in category_publications.columns:
    plt.plot(category_publications.index, category_publications[category], marker='o', label=category)

plt.title('Number of Publications Over Time by Category (Yearly)')
plt.xlabel('Year')
plt.ylabel('Number of Publications')
plt.legend(title="Category", loc='upper left')
plt.grid(True)
plt.show()

# Apply the fractal dimension calculation for each category
fractal_dimensions = {}

for category in category_publications.columns:
    fractal_dimensions[category] = adjusted_box_counting_fractal_dimension(category_publications[category].values)

print("Fractal Dimension for each category: "+fractal_dimensions)

# Group the data by year, category, and subcategory to ensure proper multi-indexing
subcategory_publications_fixed = data.groupby(['year', 'Category', 'SubCategory']).size().unstack(
    ['Category', 'SubCategory'], fill_value=0)


# Fractal dimension calculation using the Box-Counting method with adjustment for zero counts
def adjusted_box_counting_fractal_dimension(data):
    N = len(data)
    L = np.logspace(0.01, np.log10(N / 2), num=30, endpoint=False, base=10).astype(int)
    counts = []

    for l in L:
        count = np.sum([np.ptp(data[i:i + l]) > 0 for i in range(0, N, l)])
        # To handle zero counts, we add a small epsilon value before taking log
        counts.append(max(count, 1e-5))  # Prevent log(0)

    # Perform a linear fit to log-log data
    coeffs = np.polyfit(np.log(L), np.log(counts), 1)
    fractal_dimension = -coeffs[0]

    return fractal_dimension


# Iterate through each category and plot the subcategory time series with fractal dimension calculations
fractal_dimensions_by_subcategory = {}

for category in subcategory_publications_fixed.columns.get_level_values(0).unique():
    category_data = subcategory_publications_fixed[category]

    # Plot the time series data for each subcategory within the current category
    plt.figure(figsize=(12, 8))
    for subcategory in category_data.columns:
        plt.plot(category_data.index, category_data[subcategory], marker='o', label=subcategory)

    plt.title(f'Number of Publications Over Time for Category: {category}')
    plt.xlabel('Year')
    plt.ylabel('Number of Publications')
    plt.legend(title="SubCategory", loc='upper left')
    plt.grid(True)
    plt.show()

    # Apply the fractal dimension calculation for each subcategory
    fractal_dimensions_by_subcategory[category] = {}

    for subcategory in category_data.columns:
        fractal_dimensions_by_subcategory[category][subcategory] = adjusted_box_counting_fractal_dimension(
            category_data[subcategory].values)
print("Fractal Dimension for each subcategory within the categories : "+fractal_dimensions_by_subcategory)
