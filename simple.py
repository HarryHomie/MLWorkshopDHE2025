import pandas as pd

# Data for root, stem, and leaf size
data = {
    "Root Size": [0.1, 0.3, 0.5, 0.5, 0.6, 0.65, 0.6],
    "Stem Size": [0.2, 0.4, 0.4, 0.5, 0.4, 0.45, 0.4],
    "Leaf Size": [0.1, 0.05, 0.2, 0.2, 0.3, 0.2, 0.2]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to Excel in the same directory as the script
file_name = "Root_Stem_Leaf_Sizes.xlsx"
df.to_excel(file_name, index=False, sheet_name="Sizes")

file_name
