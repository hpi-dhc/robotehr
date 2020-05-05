import json

from flask import request

from robotehr.api import app
from robotehr.models import session
from robotehr.models.predictor import Predictor
from robotehr.api.helpers import (
    assert_response_type,
    build_response,
    sort_and_filter
)


@app.route('/api/predictors')
def get_predictors(
    sort_by=Predictor.created_at,
    response_type="json",
    **kwargs
):
    assert_response_type(response_type)
    q = session.query(Predictor)
    q = sort_and_filter(q, sort_by=sort_by, **kwargs)
    if response_type == 'object':
        return q.all()
    else:
        return build_response(q.all(), response_type)

@app.route('/api/predictor/details')
def get_predictor_details(predictor_id=None, response_type="json"):
    assert_response_type(response_type)
    predictor_id = predictor_id or request.args.get('predictor', type=int)
    predictor = Predictor.load(id=predictor_id)
    if response_type == 'object':
        return predictor
    else:
        return build_response(cohort.as_dict(), response_type)


def save_predictor(restored_model, training_configuration, comment, version, response_type="object"):
    # This method cannot be called via HTTP API
    assert response_type in ["object", "json"]
    clf = restored_model['clf']
    df = restored_model['X']
    df[training_configuration.target] = restored_model['y']
    predictor = Predictor.persist(df, clf, training_configuration, comment, version)
    if response_type == "json":
        return json.dumps(predictor.to_dict())
    else:
        return predictor

@app.route('/api/predictor/predict')
def predict_outcome(
    predictor_id=None,
    features=[],
    patient_id=None,
    response_type="json"
):
    assert_response_type(response_type)
    predictor_id = predictor_id or request.args.get('predictor', type=int)
    predictor = Predictor.load(id=predictor_id)
    if not features:
        if patient_id is None:
            # special case, because patient_id could be 0 (evaluates False)
            patient_id = request.args.get('patient', type=int)
        df = predictor.get_data()
        del df[predictor.training_configuration.target]
        features = list(df.iloc[patient_id])

    clf = predictor.get_clf()
    prediction = {
        'predicted_label': clf.predict([features])[0],
        'class_probabilities': clf.predict_proba([features])[0].tolist()
    }
    return build_response(prediction, response_type)
