from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session


class DBConnectionBase:
    """
    References
    ----------
    for users who want to know the differences between Engine, Connection, Session
    https://stackoverflow.com/questions/34322471/sqlalchemy-engine-connection-and-session-difference

    """

    def __init__(self, db_engine: Engine):
        self._db_engine = db_engine

    @classmethod
    def from_db_user(cls, db_type, db_driver, host, user, password, port, db, charset='utf8'):

        engine = create_engine(f"{db_type}+{db_driver}://{user}:{password}@{host}:{port}/{db}")
        return cls(engine)

    @classmethod
    def from_full_db_path(cls, full_db_path):
        engine = create_engine(f"{full_db_path}")
        return cls(engine)

    def get_db_engine(self):
        return self._db_engine

    def get_conn(self):
        conn = self._db_engine.connect()
        return conn

    def get_conn_trans(self):
        return self.get_conn().begin()

    def get_session(self) -> Session:
        Session = sessionmaker(self._db_engine)
        Session.configure(bind=self._db_engine)
        return Session()

    def execute(self, sql):
        result_proxy = self._db_engine.execute(sql)
        return result_proxy

    def clear_all_conn(self):
        self._db_engine.dispose()
        return

    def get_one(self, sql):
        return self.execute(sql).fetchone()

    def get_all(self, sql):
        return self.execute(sql).fetchall()

    def insert_from_sql(self, sql):
        trans = self.get_conn_trans()
        try:
            trans.execute(sql)
            trans.commit()

        except:
            trans.rollback()
            print('fail to insert, rollback to previous stage')

        finally:
            trans.close()
            print('db connection closed')

        return

    def insert_from_orm(self, orm_obj):
        session = self.get_session()
        try:
            session.add(orm_obj)
            session.commit()

        except:
            session.rollback()
            print('fail to insert, rollback to previous stage')

        finally:
            session.close()
            print('db connection closed')

        return

    def update_from_orm(self, orm_obj):
        pass

    def delete_from_orm(self, orm_obj, expr: str):
        sess = self.get_session()
        try:
            sess.query(orm_obj).filter(eval(expr)).delete()
            sess.commit()

        except:
            sess.rollback()

        finally:
            sess.close()

        return
