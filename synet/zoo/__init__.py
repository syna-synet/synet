from os.path import dirname, join, isfile
def find_model_path(yaml):
    return yaml if isfile(yaml) else join(dirname(__file__), yaml)
