import pandas as pd
import gzip
import json
from typing import Iterator, Dict, Any

def parse(path: str) -> Iterator[Dict[str, Any]]:
    """Parse data from .gz file

    Args:
        path (str): path of .gz file

    Yields:
        Iterator[Dict[str, Any]]: dictionary of containing data
    """
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def get_data_frame(path: str) -> pd.DataFrame:
    """Read .gz file and convert to Pandas DataFrame

    Args:
        path (str): path of .gz file

    Returns:
        pd.DataFrame: dataset
    """
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')