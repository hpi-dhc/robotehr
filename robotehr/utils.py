import inspect
import json
import sys
from pydoc import locate

import requests


def http_post(url, data, headers={}):
    default_headers = {"Content-Type": "application/json"}
    response = requests.post(
        url, data=json.dumps(data), headers={**default_headers, **headers},
    )
    if response.status_code != 200:
        raise ValueError(
            "Request returned an error %s, the response is:\n%s"
            % (response.status_code, response.text)
        )
    return response


class FriendlyNamesConverter:
    def rename_columns(self, df):
        replacements = {}
        for column in df.columns:
            replacements[column] = self.get(column)
        return replacements

    def get(self, feature):
        # does not support time window information inside feature name yet
        if feature.startswith(('age', 'gender', 'religion', 'race')):
            return feature.replace('_', ' ').replace('.', '|')

        split_name = feature.split('__')
        if split_name[1] in [
            i[0]
            for i in inspect.getmembers(
                sys.modules['fiber.condition'],
                inspect.isclass
            )
        ]:
            aggregation = split_name[0]
            split_name = split_name[1:]
        else:
            aggregation = "time series"

        if len(split_name) == 3:
            class_name, context, code = split_name
            condition_class = locate(f'fiber.condition.{class_name}')
            description = self.get_description(condition_class, code, context)
        else:
            class_name, description = split_name

        return f'{class_name} | {description.capitalize()} ({aggregation})'

    def get_description(self, condition_class, code, context):
        return condition_class(
            code=code,
            context=context
        ).patients_per(
            condition_class.description_column
        )[
            condition_class.description_column.name.lower()
        ].iloc[0]
