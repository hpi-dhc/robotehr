import os
import time
import uuid

import pandas as pd
from fiber import Cohort as FiberCohort
from fiber.config import OCCURRENCE_INDEX
from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship

from robotehr.config import BASE_PATH
from robotehr.models import Base, session

class Cohort(Base):
    __tablename__ = 'cohort'
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True)
    comment = Column(String)
    version = Column(String)
    path = Column(String)
    created_at = Column(Float)
    fiber_version = Column(String)

    onset_dfs = relationship("OnsetDataFrame", back_populates="cohort")

    def get_fiber(self):
        return FiberCohort.from_json(self.path)

    @classmethod
    def persist(cls, cohort, **kwargs):
        # persist cohort on disk
        identifier = str(uuid.uuid4())
        base_path = f"{BASE_PATH}/cohorts"
        os.makedirs(base_path, exist_ok=True)
        path = f"{base_path}/{identifier}.cohort"
        cohort_dict = cohort.to_json(path, **kwargs)

        obj = cls(
            comment=cohort_dict["comment"],
            version=cohort_dict["version"],
            created_at=cohort_dict["createdAt"],
            fiber_version=cohort_dict["fiberVersion"],
            path=path
        )
        session.add(obj)
        session.commit()
        return obj

    @classmethod
    def load(cls, id):
        obj = session.query(cls).filter_by(id=id).first()
        return obj

    @classmethod
    def load_fiber(cls, id):
        obj = cls.load(id)
        return obj.get_fiber()


class OnsetDataFrame(Base):
    __tablename__ = 'onset_df'
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True)
    comment = Column(String)
    version = Column(String)
    created_at = Column(Float)
    path = Column(String)
    cohort_id = Column(Integer, ForeignKey('cohort.id'))

    cohort = relationship("Cohort", back_populates="onset_dfs")

    @classmethod
    def persist(cls, df, comment, version, cohort):
        # persist df on disk
        identifier = str(uuid.uuid4())
        base_path = f"{BASE_PATH}/onset-df"
        os.makedirs(base_path, exist_ok=True)
        path = f"{base_path}/{identifier}.df.gz"
        df.to_csv(path, index=False)

        obj = cls(
            comment=comment,
            version=version,
            created_at=time.time(),
            path=path,
            cohort=cohort
        )
        session.add(obj)
        session.commit()
        return obj

    def get_df(self, target=''):
        df = pd.read_csv(self.path)
        df.medical_record_number = df.medical_record_number.astype(str)
        if target:
            columns = list(OCCURRENCE_INDEX) + [target]
        else:
            columns = df.columns
        return df[columns]

    @classmethod
    def load(cls, id):
        obj = session.query(cls).filter_by(id=id).first()
        return obj

    @classmethod
    def load_df(cls, id, target=''):
        obj = cls.load(id)
        return obj.get_df(obj.id, target)
