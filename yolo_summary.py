"""
Create a summary from YOLOv5 txt predictions

Usage::

    python yolo_summary.py <label_directory> <label0,label1,...> <output.csv> <min_confidence>

Labels, e.g.: 'pentagram,star_of_david,keys_of_heaven,flag'
"""
import glob
import sys
import tempfile
import os
import pandas as pd

label_directory = sys.argv[1]
labels = sys.argv[2].split(',')
output_file = sys.argv[3]
min_confidence = float(sys.argv[4])

header = {}
for i, l in enumerate(labels):
    header[i] = l + "_pred"

raw_list = f"{label_directory}.txt"
# Find all .txt files in the label directory
label_files = glob.glob(label_directory+'/**/*.txt', recursive=True)
with open(raw_list, 'w') as temp_file:
    # For each file, print the lines prepended by the filename or frame number
    for lfile in label_files:
        with open(lfile, 'r') as lf:
            for line in lf:
                temp_file.write(lfile + " " + line)

# Summarise the recognitions by frame and class, taking the highest confidence
# for each
raw_df = pd.read_csv(raw_list, 
                     sep='\s+',
                     header=None,
                     names=['filename', 'class', 'x', 'y', 'width', 'height', 'confidence'],
                     index_col=False,
                     dtype={'filename': str,
                            'class': int,
                            'x': float,
                            'y': float,
                            'width': float,
                            'height': float,
                            'confidence': float})
# Remove the directory name from the filenames
raw_df.loc[:, 'filename'] = raw_df['filename'].str.replace(label_directory + '/', '')
raw_df.loc[raw_df['confidence'] < min_confidence, 'confidence'] = 0
# Group rows
print(raw_df.head())
piv = pd.pivot_table(raw_df, values='confidence', index='filename', columns='class', aggfunc=max, fill_value=0)
piv.rename(columns=header, inplace=True)
print(piv.head())
piv.to_csv(output_file)
