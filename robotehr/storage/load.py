# import pickle

# import pandas as pd
from fiber import Cohort as FiberCohort
from robotehr.models import session
from robotehr.models.fiber import Cohort as CohortModel

# def load_features(features_run_uuid, config_type, observation_window):
#     stmt = features_table.select().where(
#         (features_table.c.run_uuid == features_run_uuid)
#         & (features_table.c.type == config_type)
#         & (features_table.c.window_start == str(observation_window[0]))
#         & (features_table.c.window_end == str(observation_window[1]))
#     )
#     db_result = engine.execute(stmt).fetchone()
#     path = db_result[features_table.c.path]
#     return pickle.load(open(path, "rb"))


# def load_trainings():
#     stmt = models_table.select()
#     return pd.read_sql(stmt, engine)


# def load_evaluation(training_uuid):
#     stmt = models_table.select().where(
#         models_table.c.training_uuid == training_uuid
#     )
#     return pd.read_sql(stmt, engine)


# def load_model(model_uuid):
#     base_path = f"{BASE_PATH}/models"
#     path = f"{base_path}/{model_uuid}.model"
#     return pickle.load(open(path, "rb"))


# def load_training_data(data_uuid):
#     base_path = f"{BASE_PATH}/training-data"
#     path = f"{base_path}/{data_uuid}.train"
#     return pickle.load(open(path, "rb"))
