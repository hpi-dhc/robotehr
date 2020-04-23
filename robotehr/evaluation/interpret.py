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
