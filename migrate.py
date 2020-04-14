# noqa

if __name__ == "__main__":
    from robotehr.models import Base, engine
    from robotehr.models.fiber import *
    from robotehr.models.training import *
    from robotehr.models.data import *

    Base.metadata.create_all(engine)
    print("Database schema update complete ...")
