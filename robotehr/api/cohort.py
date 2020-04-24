from flask import request


from robotehr.models import engine, session
from robotehr.models.cohort import Cohort
from robotehr.api import app
from robotehr.api.helpers import assert_response_type, build_response, sort_and_filter


@app.route('/api/cohort/details')
def get_cohort_details(cohort_id=None, response_type="json"):
    assert_response_type(response_type)
    cohort_id = cohort_id or request.args.get('cohort', type=int)
    assert cohort_id is not None
    cohort = Cohort.load(id=cohort_id)
    return build_response(cohort.as_dict(), response_type)


@app.route('/api/cohorts')
def get_cohorts(
    sort_by=Cohort.created_at,
    response_type="json",
    **kwargs
):
    assert_response_type(response_type)
    q = session.query(Cohort)
    q = sort_and_filter(q, sort_by=sort_by, **kwargs)
    return build_response(q.all(), response_type)
