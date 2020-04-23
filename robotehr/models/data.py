import math
import os
import time
import uuid

import pandas as pd
from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship

from robotehr.config import BASE_PATH
from robotehr.models import Base, session
from robotehr.models.cohort import Cohort


class Feature(Base):
    __tablename__ = "feature"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    feature_pipeline_id = Column(Integer, ForeignKey("feature_pipeline.id"))
    feature_type = Column(String)
    window_start = Column(Integer)
    window_end = Column(Integer)
    min_threshold = Column(Float)
    path = Column(String)
    condition = Column(String)

    feature_pipeline = relationship("FeaturePipeline", back_populates="features")

    @classmethod
    def persist(cls, df, window, min_threshold, feature_type, condition, feature_pipeline, **kwargs):
        # persist df on disk
        identifier = str(uuid.uuid4())
        base_path = f"{BASE_PATH}/features"
        os.makedirs(base_path, exist_ok=True)
        path = f"{base_path}/{identifier}.df.gz"
        df.to_csv(path, index=False)

        obj = cls(
            feature_pipeline=feature_pipeline,
            feature_type=feature_type,
            path=path,
            condition=condition.__class__.__name__,
            window_start=window[0],
            window_end=window[1],
            min_threshold=min_threshold,
        )
        session.add(obj)
        session.commit()
        return obj

    @classmethod
    def load(cls, id):
        obj = session.query(cls).filter_by(id=id).first()
        return obj


class FeaturePipeline(Base):
    __tablename__ = "feature_pipeline"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    comment = Column(String)
    version = Column(String)
    start_time = Column(Float)
    end_time = Column(Float)
    cohort_id = Column(Integer, ForeignKey("cohort.id"))

    cohort = relationship("Cohort")
    features = relationship("Feature", back_populates="feature_pipeline")

    @classmethod
    def persist(cls, comment, version, cohort):
        obj = cls(
            comment=comment,
            version=version,
            start_time=time.time(),
            end_time=math.inf,
            cohort=cohort,
        )
        session.add(obj)
        session.commit()
        return obj

    @classmethod
    def load(cls, id):
        obj = session.query(cls).filter_by(id=id).first()
        return obj

    def update_runtime(self, end_time):
        self.end_time = end_time
        session.commit()

    def end_pipeline(self):
        self.update_runtime(time.time())

    def get_features(self, train_config):
        numeric_feature_objs = session.query(Feature).filter_by(
            feature_pipeline=self,
            window_start=train_config.window_start_numeric,
            window_end=train_config.window_end_numeric,
            feature_type=train_config.feature_type_numeric
        ).all()
        numeric_feature_dfs = [pd.read_csv(f.path) for f in numeric_feature_objs]

        occurring_feature_objs = session.query(Feature).filter_by(
            feature_pipeline=self,
            window_start=train_config.window_start_occurring,
            window_end=train_config.window_end_occurring,
            feature_type=train_config.feature_type_occurring
        ).all()
        occurring_feature_dfs = [pd.read_csv(f.path) for f in occurring_feature_objs]

        return numeric_feature_dfs, occurring_feature_dfs

    @property
    def runtime(self):
        return self.end_time - self.start_time

    def as_dict(self):
        return {
            'id': self.id,
            'comment': self.comment,
            'version': self.version,
            'start_time': self.start_time,
            'runtime': str(self.runtime),
            'cohort_id': self.cohort_id
        }
