import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Path to your new feature file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
features_path = os.path.join(BASE_DIR, "data", "processed", "csv", "train_features.csv")

# Load the data
df = pd.read_csv(features_path)

# Set the visual style
plt.figure(figsize=(12, 6))
sns.boxplot(x='genre', y='spectral_centroid_mean', data=df)

plt.title('Spectral Centroid (Brightness) by Genre')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()