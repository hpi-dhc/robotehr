import pandas as pd
import shap
import morpher.config
from morpher.jobs import Explain
from morpher.plots import plot_feat_importances
from sklearn.model_selection import train_test_split

from robotehr.api.training import get_training_configuration
from robotehr.api.predictor import get_predictor_details


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


def calculate_shap_values(predictor_id):
    predictor = get_predictor_details(
        predictor_id=predictor_id,
        response_type="object"
    )
    X_train, X_test, _, _ = train_test_split(
        predictor.get_features(),
        predictor.get_targets(),
        test_size=0.2,
        random_state=0
    )
    explainer = shap.KernelExplainer(predictor.clf.predict_proba, X_train, link="logit")
    shap_values = explainer.shap_values(X_test, nsamples=100)
    return {
        'X_test': X_test,
        'shap_values': shap_values,
        'explainer': explainer
    }


def shap_plot(shap_values, X_test, **kwargs):
    shap.summary_plot(
        shap_values[1],
        X_test,
        **kwargs
    )


def calculate_lime_values(predictor_id, num_features=None):
    predictor = get_predictor_details(predictor_id=predictor_id, response_type="object")
    explanations = global_explanation(predictor, num_features=30, explainers=[morpher.config.explainers.LIME])

    if num_features:
        explanations_ = {}
        for i in list(explanations[morpher.config.explainers.LIME])[:num_features]:
            explanations_[i] = explanations[morpher.config.explainers.LIME][i]
        return explanations_
    else:
        return explanations


def plot_lime_values(explanations, friendly_names, ax, fig=None, filename=None):
    plot_feat_importances(
        explanations,
        friendly_names=friendly_names,
        title="LIME Feature Importance",
        ax=ax
    )
    if fig and filename:
        fig.savefig(filename, bbox_inches='tight')
