import pandas as pd
import re
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import numpy as np
import math


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# Function to preprocess text using nltk
def preprocess_text(textData):
    textData = re.sub(r'[^a-zA-Z\s]', '', textData, re.I | re.A)
    textData = textData.lower()
    tokens = [word for word in textData.split() if word not in stop_words]
    return ' '.join(tokens)

filePath = 'Data/CleanedDataForAnalysis.xlsx'
data = pd.read_excel(filePath, sheet_name='Sheet1')

# Filter the data for the 'stat' category and subcategories 'stat.TH' and 'stat.ML'
stat_data = data[data['Category'] == 'stat']
group_TH = stat_data[stat_data['SubCategory'] == 'stat.TH']
group_ML = stat_data[stat_data['SubCategory'] == 'stat.ML']

# Preprocess the summaries for both subcategories
summaries_sub_TH = group_TH['summary'].astype(str).apply(preprocess_text)
summaries_sub_ML = group_ML['summary'].astype(str).apply(preprocess_text)

# Create a single document-term matrix for both subcategories
vectorizer_sub = CountVectorizer(min_df=1, stop_words=None)
X_sub_TH = vectorizer_sub.fit_transform(summaries_sub_TH)
X_sub_ML = vectorizer_sub.fit_transform(summaries_sub_ML)

# Apply LDA for both subcategories
lda_sub_TH = LatentDirichletAllocation(n_components=3, random_state=43)
lda_sub_ML = LatentDirichletAllocation(n_components=3, random_state=43)

lda_sub_TH.fit(X_sub_TH)
lda_sub_ML.fit(X_sub_ML)

# Extract words and topics for both subcategories
words_TH = vectorizer_sub.get_feature_names_out()
topics_sub_TH = lda_sub_TH.components_

words_ML = vectorizer_sub.get_feature_names_out()
topics_sub_ML = lda_sub_ML.components_

# Generate word-topic network for both subcategories
G = nx.Graph()

def add_to_graph(G, words, topics_sub, subcategory):
    vocab_size = len(words)  # Get the size of the vocabulary
    for topic_idx, topic in enumerate(topics_sub):
        # Ensure we do not exceed the bounds of the vocabulary
        top_word_indices = [i for i in topic.argsort()[:-11:-1] if i < vocab_size]  # Top 10 words for each topic
        top_words = [words[i] for i in top_word_indices]
        for word in top_words:
            # Assign the topic and subcategory to the node if it doesn't already exist
            if word not in G:
                G.add_node(word, topic=topic_idx, subcategory=subcategory)
        for i in range(len(top_words)):
            for j in range(i + 1, len(top_words)):
                if G.has_edge(top_words[i], top_words[j]):
                    G[top_words[i]][top_words[j]]['weight'] += 1
                else:
                    G.add_edge(top_words[i], top_words[j], weight=1)


# Add nodes and edges for both subcategories
add_to_graph(G, words_TH, topics_sub_TH, 'stat.TH')
add_to_graph(G, words_ML, topics_sub_ML, 'stat.ML')

# Filter the graph to remove outliers (nodes with few connections)
degree_threshold = 1
filtered_nodes = [node for node, degree in G.degree() if degree > degree_threshold]
G_filtered = G.subgraph(filtered_nodes)

pos = nx.spring_layout(G_filtered, k=1.2, iterations=50)

# Color nodes and edges differently based on cross-subcategory relationships
node_colors = []
edge_colors = []
for n in G_filtered:
    if G_filtered.nodes[n]['subcategory'] == 'stat.TH':
        node_colors.append(plt.cm.Blues(G_filtered.nodes[n]['topic'] / 3.0))
    else:
        node_colors.append(plt.cm.Reds(G_filtered.nodes[n]['topic'] / 3.0))

# Assign edge colors based on whether it's a within-category or cross-category connection
for u, v in G_filtered.edges():
    if G_filtered.nodes[u]['subcategory'] == G_filtered.nodes[v]['subcategory']:
        if G_filtered.nodes[u]['subcategory'] == 'stat.TH':
            edge_colors.append('blue')
        else:
            edge_colors.append('red')
    else:
        edge_colors.append('purple')

# Draw nodes with smaller size based on degree (make the nodes smaller for better visibility)
nx.draw_networkx_nodes(G_filtered, pos, node_color=node_colors,
                       node_size=[G_filtered.degree(n) * 200 for n in
                                  G_filtered])

# Draw edges with width based on weight and color for cross-subcategory connections
nx.draw_networkx_edges(G_filtered, pos, edge_color=edge_colors, alpha=0.6,
                       width=[G_filtered[u][v]['weight'] * 2 for u, v in G_filtered.edges()])

# Draw labels for the nodes with reduced font size
nx.draw_networkx_labels(G_filtered, pos, font_size=10)  # Reduced font size for better visibility

# Add a legend to distinguish between the two subcategories
plt.text(1, 1, 'stat.TH = Blue', horizontalalignment='right', verticalalignment='top', fontsize=10, color='blue')
plt.text(1, 0.95, 'stat.ML = Red', horizontalalignment='right', verticalalignment='top', fontsize=10, color='red')

# Title of the graph for better clarity
plt.title(f"Comparison of Word-Topic Networks for stat.TH and stat.ML", fontsize=14)
plt.show()

#Fractal Dimension Calculation
def fractal_dimension(G):
    def box_count(radius):
        num_boxes = 0
        nodes_covered = set()
        for node in G:
            if node not in nodes_covered:
                # Use BFS/DFS to find nodes within the box radius
                neighbors = nx.single_source_shortest_path_length(G, node, radius)
                nodes_covered.update(neighbors.keys())
                num_boxes += 1
        return num_boxes

    radii = range(1, int(math.log(len(G.nodes()))) + 1)
    box_counts = [box_count(r) for r in radii]

    # Fit the log-log relationship to find the fractal dimension
    log_radii = np.log(radii)
    log_box_counts = np.log(box_counts)
    slope, intercept = np.polyfit(log_radii, log_box_counts, 1)

    return -slope

# Separate the subgraphs for stat.TH and stat.ML
G_TH = G_filtered.subgraph([n for n in G_filtered if G_filtered.nodes[n]['subcategory'] == 'stat.TH'])
G_ML = G_filtered.subgraph([n for n in G_filtered if G_filtered.nodes[n]['subcategory'] == 'stat.ML'])

# Calculate and print the fractal dimension for stat.TH and stat.ML subgraphs
fractal_dim_TH = fractal_dimension(G_TH)
fractal_dim_ML = fractal_dimension(G_ML)

print(f"Fractal Dimension of the stat.TH Subcategory: {fractal_dim_TH}")
print(f"Fractal Dimension of the stat.ML Subcategory: {fractal_dim_ML}")
