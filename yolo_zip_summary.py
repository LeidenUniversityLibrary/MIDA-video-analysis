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
from map_crops import get_image_file
from collections import defaultdict

def frame_to_time(ser: pd.Series) -> str:
    # The index is the frame number
    secs_raw = ser.name / 25
    mins = int((secs_raw / 60) % 60)
    hrs = int((secs_raw / 3600))
    secs = round(secs_raw % 60, 3)
    return f'{hrs:02}:{mins:02}:{secs:05.2f}'


@click.command()
@click.option('-o', '--output-csv', type=click.Path(file_okay=True), help='Output file name', required=False)
@click.option('-a', '--aux-results', type=(str, click.Path(file_okay=True)), multiple=True, help='Tuple of class and auxiliary classification of crops', required=False)
@click.option('-l', '--labels', required=True, default='pentagram,star_of_david,keys_of_heaven,flag', help='Comma-separated list of labels for the numeric classes')
@click.option('-d', '--delete-labels', required=False, default='', help='Comma-separated list of labels to ignore')
@click.option('-c', '--min-confidence', type=float, default=0.0, help='Minimum confidence level of predictions to keep')
@click.argument('label_file', type=click.Path(exists=True))
def main(label_file: str, output_csv, aux_results, labels, delete_labels, min_confidence):
    """Create a table of predictions from YOLOv5 txt files in a zip file"""
    print(label_file)
    labels_list = labels.split(',')
    label_index = {}
    for i, label in enumerate(labels_list):
        label_index[label] = i
    if delete_labels is not None and len(delete_labels) > 0:
        ignored_labels = delete_labels.split(',')
        ignored_classes = [label_index[x] for x in ignored_labels]
    else:
        ignored_classes = []
    if output_csv is None:
        output_csv = label_file.replace('.zip', '.csv')

    header = {}
    for i, l in enumerate(labels_list):
        header[i] = l + "_pred"
    header['len'] = 'number_of'

    # Load classification results
    # print(aux_results)
    true_positives = {}
    for class_name, class_file in aux_results:
        aux_classifications = pd.read_csv(class_file)
        print(f'We have extra classifications of {aux_classifications.shape[0]} {class_name} crops.')
        aux_classifications = aux_classifications[aux_classifications['prediction'] >= 0.3]
        true_positives[class_name] = aux_classifications['filename'].to_list()
        print(f'{len(true_positives[class_name])} are treated as true positives.')


    raw_list = label_file.replace('.zip', '.txt')
    crop_list = label_file.replace('.zip', '_crops.txt')
    # Read all .txt files in the archive
    with open(raw_list, 'w') as temp_file, open(crop_list, 'w') as crops_file, zipfile.ZipFile(label_file, mode='r') as lz:
        # For each file, print the lines prepended by the filename or frame number
        for lfile in lz.namelist():
            if lfile[-4:] == ".txt":
                with TextIOWrapper(lz.open(lfile, 'r')) as lf:
                    # Count occurrences per class
                    occurrences = defaultdict(int)
                    for line in lf.readlines():
                        # Find the cropped image
                        line_label = labels_list[int(line[0])]
                        occurrences[line_label] += 1
                        image_filename = get_image_file(lfile, line_label, occurrences[line_label])
                        crops_file.write(f'{lfile} {occurrences[line_label]} {image_filename}\n')
                        if line_label in true_positives and image_filename not in true_positives[line_label]:
                            # print(f'False positive: {image_filename}')
                            continue
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
    # Extract the frame number from the file name
    raw_df.loc[:, 'filename'] = raw_df['filename'].str.replace(r'^.+f-(\d+)\.txt', lambda m: m.group(1), regex=True)
    raw_df.loc[:, 'filename'] = pd.to_numeric(raw_df['filename']) #* 10
    # Ignore specified classes
    raw_df = raw_df[~raw_df['class'].isin(ignored_classes)]
    # Ignore recognitions below threshold
    raw_df = raw_df[raw_df['confidence'] >= min_confidence]
    raw_df.rename(columns={'filename':'frame'}, inplace=True)
    # Group rows
    print(raw_df.head())
    piv = pd.pivot_table(raw_df, values='confidence', index='frame', columns='class', aggfunc=[max, len], fill_value=0)
    piv.rename(columns=header, inplace=True)
    piv.columns = ['_'.join(col).strip('_') for col in piv.columns.values]
    piv.loc[:, 'timestamp'] = piv.apply(frame_to_time, axis=1)
    print(piv.head())
    piv.to_csv(output_csv)

if __name__ == "__main__":
    main()
