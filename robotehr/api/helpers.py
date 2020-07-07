import pandas as pd
from flask import jsonify
from sqlalchemy import asc, desc


def assert_response_type(response_type):
    assert response_type in ["object", "pandas", "json"]


def build_response(results, response_type):
    """Builds response based on given response type"""
    if results.__class__ == list and not results[0].__class__ == dict:
        results = [item.as_dict() for item in results]
    if response_type == "object":
        return results
    if response_type == "pandas":
        return pd.DataFrame(results)
    if response_type == "json":
        return jsonify(results)


def sort_and_filter(q, sort_by=None, sort_order=None, n_rows=None):
    """Adds sqlalchemy clauses for optional sorting and filtering"""
    if sort_by:
        if sort_order == "asc":
            q = q.order_by(asc(sort_by))
        else:
            q = q.order_by(desc(sort_by))
    if n_rows:
        q = q.limit(n_rows)
    return q
