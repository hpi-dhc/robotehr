import dill
import math
import os
import time
import uuid

import pandas as pd
from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship

from robotehr.config import BASE_PATH
from robotehr.models import Base, session
from robotehr.models.fiber import Cohort, OnsetDataFrame
from robotehr.models.data import FeaturePipeline


class TrainingPipeline(Base):
    __tablename__ = 'training_pipeline'
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True)
    comment = Column(String)
    version = Column(String)
    start_time = Column(Float)
    end_time = Column(Float)
    cohort_id = Column(Integer, ForeignKey('cohort.id'))
    onset_dataframe_id = Column(Integer, ForeignKey('onset_df.id'))
    feature_pipeline_id = Column(Integer, ForeignKey('feature_pipeline.id'))
    data_loader_path = Column(String)

    cohort = relationship("Cohort")
    onset_dataframe = relationship("OnsetDataFrame")
    training_configurations = relationship("TrainingConfiguration", back_populates="training_pipeline")
    feature_pipeline = relationship("FeaturePipeline")

    @classmethod
    def create(cls, comment, version, cohort, onset_dataframe, data_loader, feature_pipeline):
        identifier = str(uuid.uuid4())
        base_path = f"{BASE_PATH}/data-loader"
        os.makedirs(base_path, exist_ok=True)
        path = f"{base_path}/{identifier}.data-loader.gz"
        data_loader.dump(path)

        obj = cls(
            comment=comment,
            version=version,
            start_time=time.time(),
            end_time=math.inf,
            cohort=cohort,
            onset_dataframe=onset_dataframe,
            data_loader_path=path,
            feature_pipeline=feature_pipeline
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

    @property
    def runtime(self):
        return self.end_time - self.start_time

    def __repr__(self):
        return f'{self.comment} - {self.version} - started {self.start_time}, finished in {self.runtime}.'

    def as_dict(self):
        return {
            'id': self.id,
            'comment': self.comment,
            'version': self.version,
            'start_time': self.start_time,
            'runtime': str(self.runtime),
            'feature_pipeline_id': self.feature_pipeline_id,
            'onset_dataframe_id': self.onset_dataframe_id,
            'cohort_id': self.cohort_id
        }


class TrainingConfiguration(Base):
    __tablename__ = 'training_configuration'
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True)
    training_pipeline_id = Column(Integer, ForeignKey('training_pipeline.id'))
    threshold_occurring = Column(Float)
    window_start_occurring = Column(Integer)
    window_end_occurring = Column(Integer)
    feature_type_occurring = Column(String)
    threshold_numeric = Column(Float)
    window_start_numeric = Column(Integer)
    window_end_numeric = Column(Integer)
    feature_type_numeric = Column(String)
    target = Column(String)

    training_pipeline = relationship("TrainingPipeline", back_populates="training_configurations")
    training_data = relationship("TrainingData", uselist=False, back_populates="training_configuration")
    training_results = relationship("TrainingResult", back_populates="training_configuration")

    @classmethod
    def persist(
        cls,
        training_pipeline,
        threshold_occurring,
        window_start_occurring,
        window_end_occurring,
        feature_type_occurring,
        threshold_numeric,
        window_start_numeric,
        window_end_numeric,
        feature_type_numeric,
        target
    ):
        obj = cls(
            training_pipeline_id=training_pipeline.id,
            threshold_occurring=threshold_occurring,
            window_start_occurring=window_start_occurring,
            window_end_occurring=window_end_occurring,
            feature_type_occurring=feature_type_occurring,
            threshold_numeric=threshold_numeric,
            window_start_numeric=window_start_numeric,
            window_end_numeric=window_end_numeric,
            feature_type_numeric=feature_type_numeric,
            target=target
        )
        session.add(obj)
        session.commit()
        return obj

    @classmethod
    def load_by_config(cls, training_pipeline_id, config):
        obj = session.query(cls).filter_by(
            training_pipeline_id=training_pipeline_id,
            **config
        ).first()
        return obj

    def __repr__(self):
        return f'{self.threshold_occurring} - [{self.window_start_occurring}, {self.window_end_occurring}] - {self.feature_type_occurring}/{self.feature_type_numeric}'

    def as_dict(self):
        return {
            'threshold_occurring': self.threshold_occurring,
            'window_start_occurring': self.window_start_occurring,
            'window_end_occurring': self.window_end_occurring,
            'threshold_numeric': self.threshold_numeric,
            'window_start_numeric': self.window_start_numeric,
            'window_end_numeric': self.window_end_numeric,
            'feature_type_occurring': self.feature_type_occurring,
            'feature_type_numeric': self.feature_type_numeric,
            'target': self.target,
        }

class TrainingData(Base):
    __tablename__ = 'training_data'
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True)
    training_configuration_id = Column(Integer, ForeignKey('training_configuration.id'))
    path = Column(String)

    training_configuration = relationship("TrainingConfiguration", uselist=False, back_populates="training_data")

    @classmethod
    def persist(cls, training_configuration, data):
        identifier = str(uuid.uuid4())
        base_path = f"{BASE_PATH}/data"
        os.makedirs(base_path, exist_ok=True)
        path = f"{base_path}/{identifier}.data.gz"
        data.to_csv(path, index=False)

        obj = cls(
            training_configuration_id=training_configuration.id,
            path=path
        )
        session.add(obj)
        session.commit()
        return obj

    def __repr__(self):
        return f'{self.path}'

    def as_dict(self):
        return {
            'path': self.path
        }


class TrainingResult(Base):
    __tablename__ = 'training_result'
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True)
    training_configuration_id = Column(Integer, ForeignKey('training_configuration.id'))
    algorithm = Column(String)
    sampler = Column(String)
    auc_roc_mean = Column(Float)
    auc_roc_std = Column(Float)
    auc_prc_mean = Column(Float)
    auc_prc_std = Column(Float)
    f1_mean = Column(Float)
    f1_std = Column(Float)
    accuracy_mean = Column(Float)
    accuracy_std = Column(Float)
    precision_mean = Column(Float)
    precision_std = Column(Float)
    recall_mean = Column(Float)
    recall_std = Column(Float)
    average_precision_mean = Column(Float)
    average_precision_std = Column(Float)

    num_features_mean = Column(Float)
    num_features_std = Column(Float)
    num_test_rows_mean = Column(Float)
    num_test_rows_std = Column(Float)
    num_train_rows_mean = Column(Float)
    num_train_rows_std = Column(Float)
    num_sampled_train_rows_mean = Column(Float)
    num_sampled_train_rows_std = Column(Float)
    incidence_rate_train_mean = Column(Float)
    incidence_rate_train_std = Column(Float)
    incidence_rate_test_mean = Column(Float)
    incidence_rate_test_std = Column(Float)
    incidence_rate_sampled_train_mean = Column(Float)
    incidence_rate_sampled_train_std = Column(Float)

    evaluation_path = Column(String)

    training_configuration = relationship("TrainingConfiguration", back_populates="training_results")

    @classmethod
    def persist(
        cls,
        training_configuration,
        metrics,
        data_statistics,
        algorithm,
        sampler
    ):
        identifier = str(uuid.uuid4())
        base_path = f"{BASE_PATH}/evaluation"
        os.makedirs(base_path, exist_ok=True)
        evaluation_path = f"{base_path}/{identifier}.evaluation"
        dill.dump(metrics[["y_true", "y_pred", "y_probs"]], open(evaluation_path, "wb"))

        obj = cls(
            training_configuration_id=training_configuration.id,
            algorithm=algorithm.__name__,
            sampler=sampler.__name__,
            auc_roc_mean=metrics.auc_roc.mean(),
            auc_roc_std=metrics.auc_roc.std(),
            auc_prc_mean=metrics.auc_prc.mean(),
            auc_prc_std=metrics.auc_prc.std(),
            f1_mean=metrics.f1.mean(),
            f1_std=metrics.f1.std(),
            accuracy_mean=metrics.accuracy.mean(),
            accuracy_std=metrics.accuracy.std(),
            precision_mean=metrics.precision.mean(),
            precision_std=metrics.precision.std(),
            recall_mean=metrics.recall.mean(),
            recall_std=metrics.recall.std(),
            average_precision_mean=metrics.average_precision.mean(),
            average_precision_std=metrics.average_precision.std(),
            num_features_mean=data_statistics.num_features.mean(),
            num_features_std=data_statistics.num_features.std(),
            num_test_rows_mean=data_statistics.num_test_rows.mean(),
            num_test_rows_std=data_statistics.num_test_rows.std(),
            num_train_rows_mean=data_statistics.num_train_rows.mean(),
            num_train_rows_std=data_statistics.num_train_rows.std(),
            num_sampled_train_rows_mean=data_statistics.num_sampled_train_rows.mean(),
            num_sampled_train_rows_std=data_statistics.num_sampled_train_rows.std(),
            incidence_rate_train_mean=data_statistics.incidence_rate_train.mean(),
            incidence_rate_train_std=data_statistics.incidence_rate_train.std(),
            incidence_rate_test_mean=data_statistics.incidence_rate_test.mean(),
            incidence_rate_test_std=data_statistics.incidence_rate_test.std(),
            incidence_rate_sampled_train_mean=data_statistics.incidence_rate_sampled_train.mean(),
            incidence_rate_sampled_train_std=data_statistics.incidence_rate_sampled_train.std(),
            evaluation_path=evaluation_path,
        )

        session.add(obj)
        session.commit()
        return obj

    @classmethod
    def columns_with_metrics(cls, undesired_columns=None):
        if not undesired_columns:
            undesired_columns=[
                'id',
                'training_configuration_id',
                'evaluation_path'
            ]
        columns = cls.__table__.columns.keys()
        return [c for c in columns if c not in undesired_columns]

    def __repr__(self):
        return f'{self.algorithm} / {self.sampler} | AUROC: {self.auc_roc_mean:.2f}, AUPRC: {self.auc_prc_mean:.2f}, F1: {self.f1_mean:.2f}'

    def as_dict(self):
        return {
            'algorithm': self.algorithm,
            'sampler': self.sampler,
            'auc_roc_mean': self.auc_roc_mean,
            'auc_prc_mean': self.auc_prc_mean,
            'f1_mean': self.f1_mean,
            'accuracy_mean': self.accuracy_mean,
            'precision_mean': self.precision_mean,
            'recall_mean': self.recall_mean,
            'average_precision_mean': self.average_precision_mean,
        }
