import json
from jsonschema import Draft4Validator


def check_json(config_json, json_scheme_path):
    """
    Fonction de vérification de la structure du json de configuration.
    :param config_json: json à vérifier
    :param json_scheme_path: chemin vers le json schéma correspondant.
    :return: booléen indiquant si le json est valide ou non
    """
    f = open(json_scheme_path)
    json_scheme = json.load(f)

    validator = Draft4Validator(json_scheme)
    status = validator.is_valid(instance=config_json)

    return status
