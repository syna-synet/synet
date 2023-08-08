from os.path import abspath, dirname, join, isfile, commonpath
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
