"""
Add scene numbers to frames with detected symbols

Usage::

    $ python match_symbols_to_scenes.py --scenes scenes.csv --symbols symbols.csv --output output.csv

"""
import pandas as pd
import pandasql as ps
import click


@click.command()
@click.option('--scenes', type=click.Path(exists=True, file_okay=True))
@click.option('--symbols', type=click.Path(exists=True, file_okay=True))
@click.option('-o', '--output', type=click.Path(file_okay=True))
def main(scenes, symbols, output):
    """
    Add scene numbers to frames with detected symbols
    """
    # Load scenes data
    scenes_dtypes = {
        "scene": "int64",
        "first_frame": "int64",
        "last_frame": "int64"
    }
    use_cols = ["scene", "first_frame", "last_frame"]
    scenes_data = pd.read_csv(scenes, dtype = scenes_dtypes, index_col="scene", usecols=use_cols)
    print(scenes_data.head())


    # Load symbols data
    symbols_data = pd.read_csv(symbols)
    print(symbols_data.head())

    # Join the scene labels with the original frame list
    join_query = '''
    select f.*, a.scene
    from symbols_data f 
    left join scenes_data a
    on f.frame >= a.first_frame and f.frame <= a.last_frame
    '''

    frames_with_labels = ps.sqldf(join_query, locals()).set_index("frame")
    print(frames_with_labels.head())
    frames_with_labels.to_csv(output)


if __name__ == "__main__":
    main()
