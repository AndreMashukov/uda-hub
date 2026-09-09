import os
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager


Base = declarative_base()


def reset_db(db_path: str, echo: bool = True):
    """Drops the existing db file. Tables are created by the caller via metadata.create_all."""
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Removed existing {db_path}")

    engine = create_engine(f"sqlite:///{db_path}", echo=echo)
    Base.metadata.create_all(engine)
    print(f"✅ Recreated {db_path} with fresh schema")


@contextmanager
def get_session(engine: Engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def model_to_dict(instance):
    """Convert a SQLAlchemy model instance to a dictionary."""
    return {
        column.name: getattr(instance, column.name)
        for column in instance.__table__.columns
    }


def chat_interface(agent, ticket_id: str):
    """Interactive loop. Short-term memory is the LangGraph thread_id checkpointer."""
    from langchain_core.messages import HumanMessage, SystemMessage
    print("UDA-Hub chat. Type quit / exit / q to stop.")
    print(f"Session thread_id={ticket_id}")
    config = {"configurable": {"thread_id": ticket_id}}
    print(
        "Tip: include a CultPass user id (for example f556c0) or email "
        "so the supervisor can load account context."
    )
    while True:
        try:
            user_input = input("User: ").strip()
        except EOFError:
            print("\nAssistant: Goodbye!")
            break
        if not user_input:
            continue
        print("User:", user_input)
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Assistant: Goodbye!")
            break
        result = agent.invoke(
            {
                "messages": [
                    SystemMessage(content=f"ThreadId: {ticket_id}"),
                    HumanMessage(content=user_input),
                ]
            },
            config=config,
        )
        print("Assistant:", result["messages"][-1].content)
        route = result.get("route")
        classification = result.get("classification") or {}
        if classification:
            print(
                f"(route={route}, issue={classification.get('issue_type')}, "
                f"confidence={classification.get('confidence')})"
            )
