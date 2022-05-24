"""
Join a sequence of tabular files that may not be exactly of the same shape

Usage::

    $ python join_results.py --output output.csv FILE...

"""
import pandas as pd
import click


@click.command()
@click.option('-o', '--output', type=click.Path(file_okay=True), required=True)
@click.argument('files', nargs=-1, type=click.Path(exists=True), required=True)
def main(output, files):
    """
    Join a sequence of tabular files that may not be exactly of the same shape

    The file names are used as the index of the concatenated table.
    """
    print(f'Reading {len(files)} files')
    frames = []
    for f in files:
        df = pd.read_csv(f, index_col='frame')
        frames.append(df)
    joined = pd.concat(frames, keys=files, names=['episode'])
    joined.fillna(0, inplace=True)
    joined.to_csv(output)


if __name__ == "__main__":
    main()
