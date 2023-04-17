from os.path import dirname, join, isfile
def find_model_path(yaml):
    """If model yaml does not exist, take it from the model zoo (this
directory)."""
    return yaml if isfile(yaml) else join(dirname(__file__), yaml)
