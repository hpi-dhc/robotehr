import os
import time
import uuid

import dill
import pandas as pd
import matplotlib.pyplot as plt
from morpher.plots import plot_explanation_heatmap
from sqlalchemy import Column, Float, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from robotehr.config import BASE_PATH
from robotehr.models import Base, session
from robotehr.models.training import TrainingConfiguration

class Predictor(Base):
    __tablename__ = "predictor"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    comment = Column(String)
    version = Column(String)
    predictor_path = Column(String)
    data_path = Column(String)
    interpretation_path = Column(String, default="")
    created_at = Column(Float)
    training_configuration_id = Column(Integer, ForeignKey('training_configuration.id'))

    training_configuration = relationship("TrainingConfiguration")

    @classmethod
    def persist(cls, df, clf, training_configuration, comment, version, explanations=None):
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

        if explanations:
            ax = plt.gca()
            plot_explanation_heatmap(explanations, top_features=20, ax=ax)
            base_path = f"{BASE_PATH}/interpretation"
            os.makedirs(base_path, exist_ok=True)
            interpretation_path = f"{base_path}/{identifier}.png"
            ax.figure.savefig(interpretation_path, dpi=300, bbox_inches="tight")
        else:
            interpretation_path = ''

        obj = cls(
            comment=comment,
            version=version,
            predictor_path=predictor_path,
            data_path=data_path,
            created_at=time.time(),
            training_configuration_id=training_configuration.id,
            interpretation_path=interpretation_path
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

    @property
    def feature_names(self):
        df = pd.read_csv(self.data_path)
        feature_names = list(df.columns)
        feature_names.remove(self.target)
        return feature_names

    def get_features(self):
        df = pd.read_csv(self.data_path)
        df.drop(columns=[self.target], inplace=True)
        return df

    def get_targets(self):
        df = pd.read_csv(self.data_path)
        return df[self.target]

    def set_interpretation(self, interpretation_path):
        self.interpretation_path = interpretation_path
        session.commit()

    @property
    def clf(self):
        return dill.load(open(self.predictor_path, 'rb'))

    @property
    def target(self):
        return self.training_configuration.target

    def __repr__(self):
        return f'{self.comment} ({self.version}) - {self.training_configuration.target}'

    def as_dict(self):
        return {
            'comment': self.comment,
            'version': self.version,
            'target': self.training_configuration.target,
            'created_at': self.created_at,
            'training_configuration': self.training_configuration.as_dict()
        }
