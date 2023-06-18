import pandas as pd

df = pd.read_json("Resumeold.json", lines=True)
print(df.head())
