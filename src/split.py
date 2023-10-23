import pandas as pd
import shutil

tile_meta = pd.read_csv("./data/tile_meta.csv")

for index, row in tile_meta.iterrows():
    print(index, row["id"], row["dataset"])
    shutil.move(f"./data/train/{row['id']}.tif", f"./data/{row['dataset']}/{row['id']}.tif")
