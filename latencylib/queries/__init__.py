from pathlib import Path

path = Path(__file__)


def select_query() -> str:
    with path.with_name('select.sql').open('r') as f:
        return f.read()


def select_datastore_multi_entities():
    with path.with_name('datastore_multi_entities.sql').open('r') as f:
        return f.read()
