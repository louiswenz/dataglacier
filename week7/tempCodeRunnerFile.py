import json
import pandas as pd

df = pd.read_json("Resumeold.json", lines=True)
print(df["annotation"][0])


# Collect unique labels
unique_labels = set()
for data in df["annotation"]:
    for entity in data:
        for lb in entity:
            unique_labels.add(lb["label"][0])

# Print the unique labels
print("Unique Labels:")
for label in unique_labels:
    print(label)
