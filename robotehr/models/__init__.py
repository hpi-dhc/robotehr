from sqlalchemy import (
    create_engine,
    MetaData,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from robotehr.config import DB_URI

engine = create_engine(DB_URI)

Base = declarative_base()

Session = sessionmaker(bind=engine)
session = Session()
