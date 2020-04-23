from morpher.jobs import Explain
from sklearn.model_selection import train_test_split

from robotehr.pipelines.supporters.restoration import restore_model


def risk_change_by_boolean_feature(df, target, trait):
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
    pipeline_id,
    config,
    algorithm,
    sampler,
    explainers,
    num_features=20
):
    results = restore_model(pipeline_id, config, algorithm, sampler)
    X = results['X']
    y = results['y']
    clf = results['clf']

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
        models={'results': clf},
        explainers=explainers
    )['results']
    return explanations
