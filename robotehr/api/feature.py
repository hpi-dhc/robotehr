from robotehr.api import app
from robotehr.api.helpers import assert_response_type, build_response, sort_and_filter
from robotehr.models import session
from robotehr.models.data import FeaturePipeline


@app.route('/api/feature/pipelines')
def get_feature_pipelines(
    sort_by=FeaturePipeline.start_time,
    response_type="json",
    **kwargs
):
    assert_response_type(response_type)
    q = session.query(FeaturePipeline)
    q = sort_and_filter(q, sort_by=sort_by, **kwargs)
    return build_response(q.all(), response_type)
