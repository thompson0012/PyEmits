from sqlalchemy.engine import Engine
from sqlmodel import create_engine, Session, SQLModel
from sqlmodel.engine.create import _FutureEngine
from typing import Union, Optional
from pyemits.common.validation import raise_if_incorrect_type, raise_if_not_all_value_contains, \
    raise_if_not_all_element_type_uniform, check_all_element_type_uniform, raise_if_value_not_contains
from typing import List


class DBConnectionBase:
    """
    References
    ----------
    for users who want to know the differences between Engine, Connection, Session
    https://stackoverflow.com/questions/34322471/sqlalchemy-engine-connection-and-session-difference

    """

    def __init__(self, db_engine: Union[Engine, _FutureEngine]):
        self._db_engine = db_engine
        SQLModel.metadata.create_all(self._db_engine)

    @classmethod
    def from_db_user(cls, db_type, db_driver, host, user, password, port, db, charset='utf8', echo=True):
        engine = create_engine(f"{db_type}+{db_driver}://{user}:{password}@{host}:{port}/{db}", echo=echo)
        return cls(engine)

    @classmethod
    def from_full_db_path(cls, full_db_path, echo=True):
        engine = create_engine(f"{full_db_path}", echo=echo)
        return cls(engine)

    def get_db_engine(self):
        return self._db_engine

    def execute(self, sql, always_commit=False, fetch: Optional[Union[int, str]] = None):

        with Session(self._db_engine) as session:
            q = session.execute(sql)
            if always_commit:
                session.commit()

            if fetch is not None:
                raise_if_incorrect_type(fetch, (int, str))
                if isinstance(fetch, int):
                    if fetch == 1:
                        return q.fetchone()
                    elif fetch > 1:
                        return q.fetchmany(fetch)
                elif isinstance(fetch, str):
                    if fetch == 'all':
                        return q.fetchall()
                    raise ValueError
            return q

    def get_db_inspector(self):
        from sqlalchemy import inspect
        inspector = inspect(self._db_engine)
        return inspector

    def get_schemas(self, schemas='all', tables='all'):
        inspector = self.get_db_inspector()

        from collections import defaultdict
        schema_containers = defaultdict(dict)

        schemas = _validate_schema_names(inspector, schemas)

        return _get_schemas(inspector, schema_containers, schemas, tables)

    def get_tables_names(self):
        inspector = self.get_db_inspector()
        return inspector.get_table_names()


def _get_schemas(inspector, schema_containers, schemas: Union[str, List[str]], tables: Union[str, List[List[str]], List[str]]):
    schema_list = _validate_schema_names(inspector, schemas)
    if check_all_element_type_uniform(tables, list):
        for schema, table in zip(schema_list, tables):
            table_names = _validate_table_names(inspector, schema, table)
            for sub_table_names in table_names:
                schema_containers[schema][sub_table_names] = inspector.get_columns(sub_table_names, schema=schema)
        return schema_containers

    elif check_all_element_type_uniform(tables, str) or tables == 'all':
        for schema in schema_list:
            table_names = _validate_table_names(inspector, schema, tables)
            for table_name in table_names:
                schema_containers[schema][table_name] = inspector.get_columns(table_name, schema=schema)

        return schema_containers

    raise ValueError


def _validate_schema_names(inspector, schemas: List[str]):
    if schemas == 'all':
        return inspector.get_schema_names()

    if isinstance(schemas, list):
        raise_if_not_all_value_contains(schemas, inspector.get_schema_names())
        return schemas
    raise ValueError('schemas must be "all" or a list of string')


def _validate_table_names(inspector, schema: str, tables: List[str]):
    if tables == 'all':
        return inspector.get_table_names(schema=schema)

    if isinstance(tables, list):
        if check_all_element_type_uniform(tables, str):
            raise_if_value_not_contains(tables, inspector.get_table_names(schema=schema))
            return tables
        elif check_all_element_type_uniform(tables, list):
            for sub_tab in tables:
                print(sub_tab, inspector.get_table_names(schema=schema))
                raise_if_value_not_contains(sub_tab, inspector.get_table_names(schema=schema))
            return tables
    raise ValueError('tables name are not existed in database, pls verify')
