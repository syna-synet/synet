from os.path import dirname, join, isfile
def find_model_path(chip, yaml):
    return yaml if isfile(yaml) else join(dirname(__file__), chip, yaml)
