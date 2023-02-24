import json
from pseudo import load_doccano_corpus, load_custom_corpus, train_and_evaluate_model, detect_entities, evaluate_model, \
    replace_entities, write_doccano_format
from check_config import check_json

train_corpus_path = 'data/samples/sample.jsonl'
test_corpus_path = 'data/samples/sample.jsonl'
noms_prenoms_path = "data/datasets/noms_prenoms.csv"
address_path = "data/datasets/adresses_sample.csv"
car_path = "data/datasets/cars_sample.csv"
societes_path = "data/datasets/societes.csv"
csv_paths = {"noms": noms_prenoms_path, "adresses": address_path, "vehicules": car_path, "societes": societes_path}

# ENTRAINEMENT

# Chargement et vérification du fichier de configuration
path = 'data/training_config.json'
scheme_path = 'data/training_config_schema.json'
with open(path) as json_file:
    parametres = json.load(json_file)
assert check_json(parametres, scheme_path), "json d'entrée invalide."

# Chargement d'un corpus doccano annoté
[train_corpus, test_corpus] = load_doccano_corpus([train_corpus_path, test_corpus_path], parametres["labels_format"])

# Test d'entrainement
train_and_evaluate_model(parametres, train_corpus, test_corpus)

# DETECTION ET REMPLACEMENT D'ENTITES

# Chargement et vérification du fichier de configuration
path = 'data/detect_config.json'
scheme_path = 'data/detect_config_schema.json'
with open(path) as json_file:
    parametres = json.load(json_file)
assert check_json(parametres, scheme_path), "json d'entrée invalide."

# Chargement d'un ensemble de textes non annotés sous forme de liste de chaines de caractères
# (inadapté à un entrainement de modèle)
texte1 = "C’est une épave mythique. Celle de l’Endurance, le navire de l’explorateur britannique Ernest Shackleton, " \
         "brisé par les glaces en 1915 au large de l’Antarctique. Elle a été retrouvée dans la mer de Weddell par " \
         "3 000 mètres de fond, ont annoncé ses découvreurs, mercredi 9 mars."
texte2 = "Le protocole sanitaire en entreprise cessera de s’appliquer à partir de lundi 14 mars, date à laquelle le " \
         "port du masque ne sera plus obligatoire dans les lieux fermés, a annoncé mardi 8 mars sur LCI la ministre " \
         "du travail, Elisabeth Borne."
test_corpus = load_custom_corpus([texte1, texte2], parametres["labels_format"])

# Test de détection d'entités
white_space_token = "</w>"
corpus_annote = detect_entities(parametres, test_corpus)

# Remplacement des entités
corpus_pseudonymise = replace_entities(corpus_annote, parametres["entities"], csv_paths, "predicted_labels",
                                       parametres["labels_format"], white_space_token)

# Ecriture du fichier au format doccano
output_corpus_path = 'data/test/corpus_annote.jsonl'
write_doccano_format(corpus_pseudonymise, white_space_token, output_corpus_path)

# EVALUATION

# Chargement et vérification du fichier de configuration
path = 'data/evaluation_config.json'
scheme_path = 'data/evaluation_config_schema.json'
with open(path) as json_file:
    parametres = json.load(json_file)
assert check_json(parametres, scheme_path), "json d'entrée invalide."

# Chargement d'un corpus de test
[test_corpus] = load_doccano_corpus([test_corpus_path], parametres["labels_format"])

# Test d'évaluation d'un modèle
evaluate_model(parametres, test_corpus)
