import os
import time
import uuid

import dill
import pandas as pd
from sqlalchemy import Column, Float, Integer, String

from robotehr.config import BASE_PATH
from robotehr.models import Base, session

class Predictor(Base):
    __tablename__ = "predictor"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    comment = Column(String)
    version = Column(String)
    predictor_path = Column(String)
    data_path = Column(String)
    created_at = Column(Float)
    target = Column(String)

    @classmethod
    def persist(cls, df, clf, comment, version, target):
        # persist df on disk
        identifier = str(uuid.uuid4())
        base_path = f"{BASE_PATH}/prediction-data"
        os.makedirs(base_path, exist_ok=True)
        data_path = f"{base_path}/{identifier}.df.gz"
        df.to_csv(data_path, index=False)

        # persist clf on disk
        base_path = f"{BASE_PATH}/predictors"
        os.makedirs(base_path, exist_ok=True)
        predictor_path = f"{base_path}/{identifier}.clf"
        dill.dump(clf, open(predictor_path, 'wb'))

        obj = cls(
            comment=comment,
            version=version,
            predictor_path=predictor_path,
            data_path=data_path,
            target=target,
            created_at=time.time()
        )
        session.add(obj)
        session.commit()
        return obj

    @classmethod
    def load(cls, id):
        obj = session.query(cls).filter_by(id=id).first()
        return obj

    def get_data(self):
        return pd.read_csv(self.data_path)

    def get_features(self):
        df = pd.read_csv(self.data_path)
        features = list(df.columns)
        features.remove(self.target)
        return features

    def get_clf(self):
        return dill.load(open(self.predictor_path, 'rb'))

    def __repr__(self):
        return f'{self.comment} ({self.version}) - {self.target}'

    def as_dict(self):
        return {
            'comment': self.comment,
            'version': self.version,
            'target': self.target,
            'created_at': self.created_at
        }
