from os import listdir
from os.path import abspath, dirname, join, isfile, commonpath
from urllib import request


WEIGHT_URL_ROOT = "http://profiler/"


def in_zoo(model, backend):
    """Return True if model refers to something in the SyNet zoo."""
    # check if absolute path to model in the zoo was given
    if isfile(model):
        return dirname(__file__) == commonpath((__file__, abspath(model)))
    # otherwise check if name is relative to zoo dir.
    return isfile(join(dirname(__file__), backend, model))


def get_config(model, backend):
    """Return the path to a model.  Check the zoo if necessary."""
    if isfile(model):
        return model
    return join(dirname(__file__), backend, model)


def get_weights(model, backend):
    if isfile(model):
        return model
    with request.urlopen(join(WEIGHT_URL_ROOT, backend, model)) as remotefile:
        with open(model, 'wb') as localfile:
            localfile.write(remotefile.read())
    return model


def get_configs(backend):
    return listdir(join(dirname(__file__), backend))
