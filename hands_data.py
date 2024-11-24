import deeplake

# Load the dataset
ds = deeplake.load('hub://activeloop/11k-hands')

# Print the available keys in the dataset
print(ds.keys())
