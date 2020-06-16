import pandas as pd
from morpher.jobs import Explain
from sklearn.model_selection import train_test_split

from robotehr.api.training import get_training_configuration


def static_risk_change_analysis(
    pipeline_id,
    config,
):
    target = config['target']
    tc = get_training_configuration(
        pipeline_id=pipeline_id,
        response_type="object",
        config=config
    )
    data = pd.read_csv(tc.training_data.path)
    # TODO: apply transformation here
    changes = []
    for trait in data.columns:
        if trait != target:
            changes.append(
                _risk_change_by_boolean_feature(
                    data, target, trait
                )
            )

    return pd.DataFrame(changes)


def _risk_change_by_boolean_feature(df, target, trait):
    selection = df[df[trait] > 0]
    trait_incidence_rate = selection[target].sum() / len(selection)

    selection = df[df[trait] <= 0]
    no_trait_incidence_rate = selection[target].sum() / len(selection)

    return {
        "trait": trait,
        "trait_incidence_rate": trait_incidence_rate,
        "no_trait_incidence_rate": no_trait_incidence_rate,
        "change": trait_incidence_rate / no_trait_incidence_rate,
    }


def global_explanation(
    predictor,
    explainers,
    num_features=20,
):
    X = predictor.get_features()
    y = predictor.get_targets()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2
    )
    X_train['target'] = y_train
    X_test['target'] = y_test

    explanations = Explain().execute(
        data=X_train,
        exp_kwargs={
            'test': X_test,
            'num_features': num_features,
        },
        target='target',
        models={'results': predictor.clf},
        explainers=explainers
    )['results']
    return explanations