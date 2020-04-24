# noqa

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv('.flaskenv')

    from robotehr.models import Base, engine
    from robotehr.models.cohort import *
    from robotehr.models.training import *
    from robotehr.models.data import *
    from robotehr.models.predictor import *

    Base.metadata.create_all(engine)
    print("Database schema update complete ...")
