"""
Create a table of predictions from YOLOv5 txt files in a zip file

Usage::

    python yolo_zip_summary.py [options] <zip_file>

Labels, e.g.: 'pentagram,star_of_david,keys_of_heaven,flag'
"""
from io import TextIOWrapper
import zipfile
import click
import pandas as pd

def frame_to_time(ser: pd.Series) -> str:
    # The index is the frame number
    secs_raw = ser.name / 25
    mins = int((secs_raw / 60) % 60)
    hrs = int((secs_raw / 3600))
    secs = round(secs_raw % 60, 3)
    return f'{hrs}:{mins}:{secs}'

@click.command()
@click.option('-o', '--output-csv', type=click.Path(file_okay=True), help='Output file name', required=False)
@click.option('-l', '--labels', required=True, default='pentagram,star_of_david,keys_of_heaven,flag')
@click.option('-c', '--min-confidence', type=float, default=0.0)
@click.argument('label_file', type=click.Path(exists=True))
def main(label_file: str, output_csv, labels, min_confidence):
    """Create a table of predictions from YOLOv5 txt files in a zip file"""
    labels_list = labels.split(',')
    if output_csv is None:
        output_csv = label_file.replace('.zip', '.csv')

    header = {}
    for i, l in enumerate(labels_list):
        header[i] = l + "_pred"

    base_name = label_file[:label_file.rindex('_')]
    print(base_name)
    raw_list = label_file.replace('.zip', '.txt')
    # Read all .txt files in the archive
    with open(raw_list, 'w') as temp_file, zipfile.ZipFile(label_file, mode='r') as lz:
        # For each file, print the lines prepended by the filename or frame number
        for lfile in lz.namelist():
            with TextIOWrapper(lz.open(lfile, 'r')) as lf:
                for line in lf.readlines():
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
    raw_df.loc[:, 'filename'] = raw_df['filename'].str.replace('labels/' + base_name + '_0f_', '', regex=False)
    raw_df.loc[:, 'filename'] = raw_df['filename'].str.replace('.txt', '', regex=False)
    raw_df.loc[:, 'filename'] = pd.to_numeric(raw_df['filename']) * 10
    raw_df = raw_df[raw_df['confidence'] >= min_confidence]
    raw_df.rename(columns={'filename':'frame'}, inplace=True)
    # Group rows
    print(raw_df.head())
    piv = pd.pivot_table(raw_df, values='confidence', index='frame', columns='class', aggfunc=max, fill_value=0)
    piv.rename(columns=header, inplace=True)
    piv.loc[:, 'timestamp'] = piv.apply(frame_to_time, axis=1)
    print(piv.head())
    piv.to_csv(output_csv)

if __name__ == "__main__":
    main()
