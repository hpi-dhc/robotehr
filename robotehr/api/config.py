import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    """Configuration to be used by the RobotEHR HTTP API"""
    SECRET_KEY = os.environ.get('SECRET_KEY')
