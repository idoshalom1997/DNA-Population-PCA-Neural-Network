"""Do French, Spanish and Turkish people differ genetically? PCA of 20,000 SNPs.

Code exported from 1_pca_populations.ipynb; see the notebook for outputs and charts.
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the paths to the files
geno_path = 'data/geno'
ind_path = 'data/ind'
snp_path = 'data/snp'

# Load the data
geno = pd.read_csv(geno_path, header=None)
ind = pd.read_csv(ind_path, header=None, names=['Population'])
snp = pd.read_csv(snp_path, header=None, names=['SNP'])

# Determine the number of explanatory variables and observations
num_explanatory_variables = geno.shape[1]
num_observations = geno.shape[0]

# Show a distribution of the dependent variable
population_distribution = ind['Population'].value_counts()

# Print the results
print(f'Number of explanatory variables: {num_explanatory_variables}')
print(f'Number of observations: {num_observations}')

# Map the original population labels to the desired labels
label_mapping = {
    'Spanish.HO_PCA': 'Spanish',
    'Turkish.HO_PCA': 'Turkish',
    'French.HO_PCA': 'French',
    'Unknown': 'Others'
}

# Apply the mapping to the population data
ind['Population'] = ind['Population'].map(label_mapping)
population_distribution = ind['Population'].value_counts()

# Define colors for each population group
colors = {
    'Spanish': 'coral',
    'Turkish': 'lightgreen',
    'French': 'navy',
    'Others': 'deeppink'
}

# Plot the distribution
plt.figure(figsize=(10, 6))
population_distribution.plot(kind='bar', color=[colors[label] for label in population_distribution.index])
plt.title('Distribution of Populations', fontsize=16)
plt.xlabel('Population', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', linewidth=0.7)
plt.tight_layout()

# Annotate bars with counts
for i, (label, count) in enumerate(population_distribution.items()):
    plt.text(i, count + 5, str(count), ha='center', fontsize=12, color=colors[label])

plt.show()

# %%
# Identify and Remove Low Variance Variables
variances = geno.var(axis=0)
low_variance_cols = variances[variances == 0].index
geno_cleaned = geno.drop(columns=low_variance_cols)

# Print the number of SNPs removed
print(f'Number of SNPs with zero variance removed: {len(low_variance_cols)}')
print(f'Number of remaining SNPs: {geno_cleaned.shape[1]}')

# Normalize the Data
geno_normalized = (geno_cleaned - geno_cleaned.mean()) / geno_cleaned.std()

# %%
# Perform SVD
U, S, VT = np.linalg.svd(geno_normalized, full_matrices=False)

# Dimensions of the components
print(f'Dimensions of U: {U.shape}')
print(f'Dimensions of S: {S.shape}')
print(f'Dimensions of VT: {VT.shape}')

# %%
import numpy as np
import pandas as pd

# Define the column names
column_names = ['SNP', 'Chromosome', 'Frequent_Allele', 'Rare_Allele']

# Read the CSV file 
snp = pd.read_csv(snp_path, header=None)
snp.columns = column_names

# Extract loadings for the first principal component
first_pc_loadings = VT[0]

# Identify the indices of the ten most influential genetic markers
most_influential_indices = np.argsort(np.abs(first_pc_loadings))[-10:][::-1]

# Map these indices to SNP names using the snp DataFrame
most_influential_snps = snp.iloc[most_influential_indices]

# Display the results
print("The ten most influential genetic markers:")
print(most_influential_snps[['SNP', 'Chromosome', 'Frequent_Allele', 'Rare_Allele']])

# %%
# Extract loadings for the first two principal components
PC1 = VT[0, :]
PC2 = VT[1, :]

# Project the data onto the first two principal components
proj_PC1 = np.dot(geno_normalized, PC1)
proj_PC2 = np.dot(geno_normalized, PC2)

# %%
# Create a DataFrame with the projections and the population labels
projections_df = pd.DataFrame({
    'PC1': proj_PC1,
    'PC2': proj_PC2,
    'Population': ind['Population']
})

# Define the populations of interest
our_populations = ['French', 'Spanish', 'Turkish']

# Filter the DataFrame to include only the Turkish, Spanish, and French populations
filtered_projections_df = projections_df[projections_df['Population'].isin(our_populations)]

# Plotting the projections on the first and second principal components for the filtered populations
plt.figure(figsize=(10, 7))

# Scatter plot for the projections with different colors for each population
colors = {'French': 'navy', 'Spanish': 'coral', 'Turkish': 'lightgreen'}
for population in filtered_projections_df['Population'].unique():
    indices = filtered_projections_df[filtered_projections_df['Population'] == population]
    plt.scatter(indices['PC1'], indices['PC2'], alpha=0.7, label=population, color=colors[population])

plt.title('Projection on PC1 & PC2')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

# Perform SVD
U, S, VT = np.linalg.svd(geno_normalized, full_matrices=False)

# Total explained variance from PCs
explained_variance = (S**2) / np.sum(S**2)
cumulative_explained_variance = np.cumsum(explained_variance)

# Plotting the cumulative explained variance 
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, 'o', markersize=2, color='magenta')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.tight_layout()
plt.show()

# Variance recovered by the initial 2 eigencomponents
variance_2_components = cumulative_explained_variance[1] * 100
print(f'Variance recovered by the initial 2 eigencomponents: {variance_2_components:.2f}%')

# %%
# Number of eigencomponents needed to explain 50% variance
num_components = np.argmax(cumulative_explained_variance >= 0.5) + 1
print(f'Number of components to explain at least 50% variance: {num_components}')

# Reconstruct the data 
U_reduced = U[:, :num_components]
S_reduced = np.diag(S[:num_components])
VT_reduced = VT[:num_components, :]
reconstructed_data = np.dot(U_reduced, np.dot(S_reduced, VT_reduced))

# Calculate the reconstruction error using the Frobenius norm
reconstruction_error = np.linalg.norm(geno_normalized - reconstructed_data, 'fro')
print(f'Reconstruction error: {reconstruction_error:.2f}')
