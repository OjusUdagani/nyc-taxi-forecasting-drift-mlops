import pandas as pd
import glob

def combineData(readPath, writePath):
    files = glob.glob(readPath)
    # Read and concatenate all at once
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # Save merged file
    df.to_parquet(writePath, index=False)
    print(f"Total Size of new combined Data Frame is: {df.size}")
