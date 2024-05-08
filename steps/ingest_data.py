import logging, os, glob, sys
from zenml import step
import pandas as pd
from typing import Tuple

@step
def ingest_data(csv_file: str, data_root: str) -> Tuple[list, int]:
    """
    A `step` to load the Artists Dataset and provide as dataframe.
    
    Args:
        csv_file: str
        data_root: str
    Returns:
        dataset: list
        n_genres: int
    """
    try:
        dataset = []
        unique_genres = []
        logging.info('Preparing to ingest dataset')
        
        df = pd.read_csv(csv_file)[['name', 'genre']]
        
        for i in range(df.shape[0]):
            img_list = glob.glob(os.path.join(data_root, df.iloc[i]['name'].replace(' ','_'), '*.jpg'))
            genre = df.iloc[i]['genre'].split(',')
            unique_genres.extend(genre)
            for img_path in img_list:
                dataset.append({"img_loc": img_path, "genre":genre})

        n_genres = len(set(unique_genres))
        logging.info('Dataset ingested successfully ')

        return dataset, n_genres
    except Exception as e:
        _, _, line = sys.exc_info()
        logging.error(f"Error in Dataset ingestion: {e}. Line No.{line.tb_lineno}")
        raise e
    
if __name__ == '__main__':
    dataset, n_genres = ingest_data(data_root='data/images/images', csv_file='data/artists.csv')
    print(dataset[200:210])
    print(len(dataset))
    print(n_genres)