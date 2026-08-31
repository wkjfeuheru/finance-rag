from finance_rag.src.infrastructure.relational_db.postgres import PostgresDatabase


def test_postgres_session_closes_database_session(monkeypatch):
    class FakeSession:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    session = FakeSession()
    database = PostgresDatabase("sqlite://")
    monkeypatch.setattr(database, "_get_session_factory", lambda: lambda: session)

    generator = database.session()
    assert next(generator) is session
    generator.close()

    assert session.closed is True
