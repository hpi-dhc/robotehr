import os
import pickle
import time
import uuid
from typing import Optional

import pandas as pd

from robotehr.config import BASE_PATH
from robotehr.models import session
from robotehr.models.fiber import Cohort, OnsetDataFrame



# def persist_extracted_features(
#     run_uuid,
#     conds,
#     configs,
#     windows,
#     results,
#     cohort_uuid,
#     config_type,
#     comment,
#     version,
#     elapsed_time,
#     min_threshold=0,
#     target=""
# ):
#     current_output = []
#     base_path = f"{BASE_PATH}/features"
#     os.makedirs(base_path, exist_ok=True)

#     for window in range(0, len(results), len(conds)):
#         meta = results[window]
#         identifier = str(uuid.uuid4())
#         path = f"{base_path}/{identifier}.features"
#         current_output.append({
#             "run_uuid": run_uuid,
#             "features_uuid": identifier,
#             "window_start": meta["window_start"],
#             "window_end": meta["window_end"],
#             "min_threshold": min_threshold,
#             "type": config_type,
#             "target": target
#         })
#         pickle.dump(
#             [x["df"] for x in results[window:(window + len(conds))]],
#             open(path, "wb")
#         )

#         stmt = features_table.insert().values(
#             features_uuid=identifier,
#             comment=comment,
#             version=version,
#             created_at=time.time(),
#             elapsed_time=elapsed_time,
#             path=path,
#             cohort_uuid=cohort_uuid,
#             window_start=meta["window_start"],
#             window_end=meta["window_end"],
#             min_threshold=min_threshold,
#             type=config_type,
#             run_uuid=run_uuid,
#             target=target
#         )
#         engine.execute(stmt)

#     return current_output


# def persist_training(
#     test,
#     train,
#     target,
#     train_sampled,
#     config,
#     algorithm,
#     model,
#     feature_run_uuid,
#     metrics,
#     sampling_method,
#     training_uuid,
#     comment,
#     version,
#     data_id
# ):
#     model_id = str(uuid.uuid4())

#     base_path = f"{BASE_PATH}/models"
#     os.makedirs(base_path, exist_ok=True)
#     model_filename = f"{base_path}/{model_id}.model"
#     pickle.dump(model, open(model_filename, "wb"))

#     base_path = f"{BASE_PATH}/training-data"
#     os.makedirs(base_path, exist_ok=True)
#     filename = f"{base_path}/{data_id}"
#     train.to_csv(f"{filename}.train.gz", index=False)
#     test.to_csv(f"{filename}.test.gz", index=False)

#     if config["window_numeric"][0].__class__.__name__ == 'int':
#         window_numeric = config["window_numeric"]
#     else:
#         window_numeric = []
#         window_numeric.append(config["window_numeric"][0][0])
#         window_numeric.append(
#             config["window_numeric"][len(config["window_numeric"]) - 1][1]
#         )

#     stmt = models_table.insert().values(
#         model_id=model_id,
#         training_uuid=training_uuid,
#         threshold_occurring=config["threshold_occurring"],
#         window_start_occurring=config["window_occurring"][0],
#         window_end_occurring=config["window_occurring"][1],
#         threshold_numeric=config["threshold_numeric"],
#         window_start_numeric=window_numeric[0],
#         window_end_numeric=window_numeric[1],
#         target=target,
#         model=algorithm.__name__,
#         sampler=sampling_method.__name__,
#         num_test_rows=num_test_rows,
#         num_train_rows=num_train_rows,
#         num_features=num_features,
#         incidence_rate_train=incidence_rate_train,
#         incidence_rate_test=incidence_rate_test,
#         num_sampled_train_rows=num_sampled_train_rows,
#         incidence_rate_sampled_train=incidence_rate_sampled_train,
#         feature_run_uuid=feature_run_uuid,
#         average_precision=metrics["average_precision"].mean(),
#         f1=metrics["f1"].mean(),
#         precision=metrics["precision"].mean(),
#         recall=metrics["recall"].mean(),
#         accuracy=metrics["accuracy"].mean(),
#         roc_auc=metrics["roc_auc"].mean(),
#         comment=comment,
#         version=version,
#         data_id=data_id
#     )

#     engine.execute(stmt)
#     return {
#         "uuid": model_id,
#         "path": model_filename
#     }
