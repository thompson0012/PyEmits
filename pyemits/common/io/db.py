from sqlalchemy.engine import Engine
from sqlmodel import create_engine, Session, SQLModel
from sqlmodel.engine.create import _FutureEngine
from typing import Union, Optional
from pyemits.common.validation import raise_if_incorrect_type


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
        engine = create_engine(f"{full_db_path}", echo=True)
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

    def get_schemas(self):
        inspector = self.get_db_inspector()
        schemas = inspector.get_schema_names()

        from collections import defaultdict
        schema_containers = defaultdict(dict)
        for schema in schemas:
            # print("schema: %s" % schema)
            for table_name in inspector.get_table_names(schema=schema):
                schema_containers[schema][table_name] = inspector.get_columns(table_name, schema=schema)

        return schema_containers

    def get_tables_names(self):
        inspector = self.get_db_inspector()
        return inspector.get_table_names()




