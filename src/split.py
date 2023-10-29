import pandas as pd
import shutil
import os

tile_meta = pd.read_csv("/home/shadowbyte/Documents/hacking-the-human-vasculature/data/tile_meta.csv")

for index, row in tile_meta.iterrows():
    print(index, row["id"], row["dataset"])
    
    src_path = f"/home/shadowbyte/Documents/hacking-the-human-vasculature/data/train/{row['id']}.tif"
    dst_path = f"/home/shadowbyte/Documents/hacking-the-human-vasculature/data/{row['dataset']}/{row['id']}.tif"
    
    # Crear el directorio destino si no existe
    if not os.path.exists(os.path.dirname(dst_path)):
        os.makedirs(os.path.dirname(dst_path))
    
    shutil.move(src_path, dst_path)

