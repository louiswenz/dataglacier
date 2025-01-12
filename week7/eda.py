import json
import pandas as pd

df = pd.read_json("Resumeold.json", lines=True)
# print(df["annotation"][0])


# Collect unique labels
unique_labels = []
# for data in df["annotation"]:
#     for entity in data:

#         unique_labels.add(entity["label"][0])


for data in df["annotation"]:
    for entity in data:
        unique_labels.extend(entity["label"])
unique_labels = set(unique_labels)
# Print the unique labels
print("Unique Labels:")
for label in unique_labels:
    print(label)
