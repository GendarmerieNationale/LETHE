import os
import random
import logging
import torch
import re
import json
import locale
import unidecode
import string
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from operator import itemgetter
from datetime import datetime, timedelta
from random import random, randint, choice, seed
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm
from faker import Faker
from transformers import get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from sklearn.metrics import precision_score as sk_precision_score, recall_score as sk_recall_score, \
    f1_score as sk_f1_score, confusion_matrix as sk_confusion_matrix

from utils import convert_to_features, switch_entity, find_sub_list, get_text_and_labels
from focal_loss import FocalLoss


class Ner:

    def __init__(self, _inputs, log_level=logging.INFO):

        self.logger = logging.getLogger(__name__)
        self.log_level = log_level

        self.inputs = _inputs
        self.pad_token_label_id = CrossEntropyLoss().ignore_index
        self.set_seed(_inputs["seed"])
        self.list_entities = _inputs["entities"]
        self.underlines = {
            ent: '#%02x%02x%02x' % (int(sns.color_palette('pastel', len(self.list_entities))[i][0] * 255),
                                    int(sns.color_palette('pastel', len(self.list_entities))[i][1] * 255),
                                    int(sns.color_palette('pastel', len(self.list_entities))[i][2] * 255))
            for i, ent in enumerate(self.list_entities)}
        self.list_regex = _inputs["regex"]
        self.max_seq_length = _inputs["max_seq_length"]
        self.per_gpu_batch_size = _inputs["per_gpu_batch_size"]
        self.model_path = _inputs["model_path"]
        self.tokenizer_path = _inputs["tokenizer_path"]
        self.labels_format = _inputs["labels_format"]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialisation des paramètres
        self.adam_epsilon, self.learning_rate, self.max_steps, self.gradient_accumulation_steps, self.num_train_epochs,\
            self.max_grad_norm, self.warmup_steps, self.weight_decay, self.white_space_token, self.loss_function, \
            self.output_dir = [None] * 11

    def evaluate_model(self, corpus):
        """
        Evaluation du modèle pour les entités précisées.
        :param corpus: DataFrame du corpus à utiliser pour l'évaluation
        """

        # Loading labels
        labels, labels_weight = self.load_labels(None)
        # Loading model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer(labels)
        # Evaluation
        eval_dataset = self.load_and_cache_texts(corpus, tokenizer, labels)

        # Save config and logs
        self.save_config_and_logs()

        model.to(self.device)
        result, _ = self.run_predict_and_eval(eval_dataset, model, tokenizer, labels, self.model_path)

    def evaluate_and_display_results(self, eval_loss, real_labels, predicted_labels, labels, no_saving, model_file):
        """
        Evalue les performances d'un modèle et sauvegarde les résultats dans le dossier du modèle.
        :param eval_loss: valeur moyenne de la fonction de perte
        :param real_labels: liste des labels correspondant aux tokens du texte
        :param predicted_labels: liste des labels prédits par l'algorithme
        :param labels: liste des différents labels possibles
        :param no_saving: booléen précisant si les résultats de l'évaluation doivent etre enregistrés ou non
        :param model_file: chemin vers le modèle évalué
        return: résultats de l'évaluation sous forme de dictionnaire
        """
        # Computes metrics
        results = self.get_scores(real_labels, predicted_labels, labels, eval_loss)
        # Displays results and saves them to a file
        # for key in sorted(results.keys()):
        #     self.logger.info("  %s = %s", key, str(results[key]))
        self.logger.info("1. results by entity\n")
        for ent in self.list_entities:
            end_of_line = "\n" if ent == self.list_entities[-1] else ""
            self.logger.info("\t%s : %s%s", ent, str(results[ent]), end_of_line)
        self.logger.info("2. global results\n")
        other_keys = set(results.keys()) - set(self.list_entities) - {"confusion_matrix"}
        for key in other_keys:
            end_of_line = "\n" if key == list(other_keys)[-1] else ""
            self.logger.info("\t%s = %s%s", key, str(results[key]), end_of_line)
        self.logger.info("3. confusion matrix\n")
        self.logger.info("\t%s\n", str(results["confusion_matrix"]))
        # Saves results
        if not no_saving:
            output_eval_file = model_file.replace('.pt', '_eval_results.txt')
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

        return results

    def extract_info_from_batch(self, tokenizer, batch, _output_probabilities, label_map, threshold=None):
        """
        Extraction des différentes informations contenues dans un batch de données.
        :param tokenizer: tokenizer du modèle
        :param batch: batch de données
        :param _output_probabilities: probabilités des différentes classes données par l'algorithme
        :param label_map: dictionnaire de correspondance entre les libellés des labels et leurs identifiants
        :param threshold: seuils associés à chaque classe. Si la probabilité de sortie dépasse ce seuil, on considère
        que l'algorithme l'a prédite meme si ce n'est pas la probabilité maximale.
        :return:
        """

        token_2_ignore = [tokenizer.sep_token, tokenizer.cls_token, tokenizer.pad_token]
        token_ids_2_ignore = [tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id]

        # Extract texts and predicted labels
        text_tokens = [[tokenizer.convert_ids_to_tokens(int(x)) for x in y] for y in batch[0]]
        labels_probabilities = _output_probabilities.detach().to("cpu").numpy()
        predicted_labels_ids = np.argmax(labels_probabilities, axis=2)

        # Using manual threshold
        if threshold is not None:
            for i, row in enumerate(labels_probabilities):
                for j, token in enumerate(row):
                    if any([x >= threshold[ind] for ind, x in enumerate(token)][1:]) and np.argmax(token) == 0:
                        _rescaled_tokens = [x if (ind != 0) and (x >= threshold[ind]) else -1000 for ind, x in
                                            enumerate(token)]
                        predicted_labels_ids[i][j] = np.argmax(_rescaled_tokens)

        predicted_labels = [[label_map[x] for x in y] for y in predicted_labels_ids]
        # Delete functional tokens
        labels_probabilities = [[", ".join([str(z) for z in y]) for y in x] for x in labels_probabilities]
        _joined = [[(x, y, z) for x, y, z in zip(text_tokens[i], predicted_labels[i], labels_probabilities[i]) if
                    x not in token_2_ignore]
                   for i in range(len(text_tokens))]
        _valid_examples = [i for i, x in enumerate(_joined) if len(x) > 0]
        _joined = [list(zip(*_joined[i])) for i in _valid_examples]
        text_tokens = [list(x[0]) for x in _joined]
        predicted_labels = [list(x[1]) for x in _joined]
        labels_probabilities = [list(x[2]) for x in _joined]
        # Extract real labels
        real_labels = [[label_map[int(x)] for x in y if x != self.pad_token_label_id] for y in batch[3]]
        real_labels = [x for i, x in enumerate(real_labels) if i in _valid_examples]
        # Extract file names
        file_tokens = [[tokenizer.convert_ids_to_tokens(int(x)) for x in y if x not in token_ids_2_ignore] for y in
                       batch[4]]
        files = ["".join([x.replace('▁', ' ') for x in y]).strip() for y in file_tokens]
        files = [x for i, x in enumerate(files) if i in _valid_examples]
        # Extract text part
        text_parts = [int(x) for x in batch[5]]
        text_parts = [x for i, x in enumerate(text_parts) if i in _valid_examples]

        return files, text_parts, text_tokens, real_labels, predicted_labels, labels_probabilities

    def find_regex_entities(self, corpus):
        """
        Détection des entités repérées par des expressions régulières et remplacement des tags correspondant par le
        label adéquat.
        param corpus: corpus de textes
        return : corpus avec nouveaux labels
        """

        func_dic = {"TIME": self.regex_time, "PHONE": self.regex_phone, "IMMAT": self.regex_immat,
                    "EMAIL": self.regex_email}

        for regex in self.list_regex:
            corpus = corpus.apply(lambda x: func_dic[regex](x), axis=1)

        return corpus

    @staticmethod
    def get_pos_class_freq(train_df):
        """
        Calcule le vecteur de poids pour la dernière couche du réseau, après l'encodeur. Le poids de chaque classe de
        sortie est inversement proportionnel à la fréquence de la classe dans le dataset d'entrainement.
        :param train_df: DataFrame du corpus d'entrainement
        :return: dictionnaire associant un poids à chaque classe
        """

        count_df = pd.Series([y for x in train_df.labels.values for y in x]).value_counts().reset_index()
        return {e[0]: e[1] for e in count_df[['index', 0]].values}

    def get_scores(self, real_labels, predicted_labels, labels, eval_loss):
        """
        Calcul des performances du modèle (f1, rappel et précision) au global et pour chaque entité.
        :param real_labels: liste des labels correspondant aux tokens du texte
        :param predicted_labels: liste des labels prédits par l'algorithme
        :param labels: liste des différents labels possibles
        :param eval_loss: valeur moyenne de la fonction de perte
        :return: dictionnaire des performances de l'algorithme.
        """

        _s_labels = list(sorted(labels))
        _flat_real_labels = [x for y in real_labels for x in y]
        _flat_predicted_labels = [x for y in predicted_labels for x in y]
        _flat_real_labels_type_only = [x.split("-")[-1] for y in real_labels for x in y]
        _flat_predicted_labels_type_only = [x.split("-")[-1] for y in predicted_labels for x in y]
        _labels_type_only = list(set([x.split("-")[-1] for x in labels if x != 'O']))
        cm = sk_confusion_matrix(_flat_real_labels, _flat_predicted_labels, labels=_s_labels)
        cm = np.concatenate((np.transpose(np.array([[''] + _s_labels])), np.concatenate((np.array([_s_labels]),
                                                                                         cm), axis=0)), axis=1)

        results = {
            "loss": eval_loss,
            "precision (entity type only)": sk_precision_score(_flat_real_labels_type_only,
                                                               _flat_predicted_labels_type_only,
                                                               labels=_labels_type_only, average='micro',
                                                               zero_division=0),
            "precision (BIO labels)": sk_precision_score(_flat_real_labels, _flat_predicted_labels,
                                                         labels=[x for x in labels if x != "O"], average='micro',
                                                         zero_division=0),
            "recall (entity type only)": sk_recall_score(_flat_real_labels_type_only, _flat_predicted_labels_type_only,
                                                         labels=_labels_type_only, average='micro', zero_division=0),
            "recall (BIO labels)": sk_recall_score(_flat_real_labels, _flat_predicted_labels,
                                                   labels=[x for x in labels if x != "O"], average='micro',
                                                   zero_division=0),
            "f1 (entity type only)": sk_f1_score(_flat_real_labels_type_only, _flat_predicted_labels_type_only,
                                                 labels=_labels_type_only, average='micro', zero_division=0),
            "f1 (BIO labels)": sk_f1_score(_flat_real_labels, _flat_predicted_labels,
                                           labels=[x for x in labels if x != "O"], average='micro', zero_division=0),
            "confusion_matrix": cm
        }

        for ent in self.list_entities:
            _preds = [1 if x == ent else 0 for x in _flat_predicted_labels_type_only]
            _reals = [1 if x == ent else 0 for x in _flat_real_labels_type_only]
            results[ent] = f"precision: {sk_precision_score(_reals, _preds, zero_division=0)}, " \
                           f"recall: {sk_recall_score(_reals, _preds, zero_division=0)}"

        return results

    def get_corpus_stats(self, corpus):
        """
        Ajoute aux logs les caractéristiques du corpus traité.
        :param corpus: DataFrame du corpus de textes
        """
        _global = f"{len(corpus)} textes dans le corpus, soit {sum([len(x) for x in corpus.text.tolist()])} tokens.\n"
        _per_entity = "Nombre d'entités:\n"
        for ent in self.list_entities:
            _per_entity += f"\t- {ent} : {[x for y in corpus.labels.tolist() for x in y].count(ent)}\n"
        self.logger.info("%s\n%s", _global, _per_entity)

    def load_model_and_tokenizer(self, labels):
        """
        Chargement du modèle et du tokenizer associé.
        :param labels: liste des différents labels possibles
        :return: modèle et tokenizer
        """

        if self.model_path.endswith(".pt"):
            if self.device.type == "cpu":
                model = torch.load(self.model_path, map_location=torch.device('cpu'))
            else:
                model = torch.load(self.model_path)

            tokenizer_path = os.path.join(self.tokenizer_path, model.__class__.__name__)

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.white_space_token = tokenizer.tokenize("le")[0].replace("le", "")
        else:
            config_file = os.path.join(self.model_path, "config.json")

            config = AutoConfig.from_pretrained(config_file)
            config.num_labels = len(labels)

            model = AutoModelForTokenClassification.from_config(config)

            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.white_space_token = tokenizer.tokenize("le")[0].replace("le", "")

        return model, tokenizer

    def load_labels(self, train_df):
        """
        Génère les labels en fonction du format choisi ainsi que leurs poids en fonction de leur fréquence
        d'apparition dans le corpus.
        :param train_df: corpus de textes d'entrainement
        :return: liste des labels et poids correspondants
        """
        if self.labels_format == "BIO":
            labels = ["O"] + [y for z in [[f"B-{x}", f"I-{x}"] for x in self.list_entities] for y in z]
        else:
            labels = ["O"] + self.list_entities

        # Les poids des différents labels sont calculés à partir de leur fréquence d'apparition.
        if (train_df is None) or (len(train_df) == 0):
            labels_weights = [1 for _ in labels]
        # Si l'on veut uniquement faire des prédictions, on peut se contenter d'un vecteur de poids constant
        else:
            freqs = self.get_pos_class_freq(train_df)
            labels_weights = np.array([freqs.get(key, None) for key in labels], dtype=np.float64)
            labels_weights = [np.nanmax(labels_weights) / x if not np.isnan([x]) else np.nanmax(labels_weights) for x in
                              labels_weights]
            labels_weights = [np.log(x) if x != 1 else x for x in labels_weights]
            labels_weights = torch.tensor(labels_weights).float()
            labels_weights = labels_weights.to(device=self.device)

        return labels, labels_weights

    def load_and_cache_texts(self, corpus_df, tokenizer, labels):
        """
        Charge les différents textes du corpus dans un TensorDataset.
        :param corpus_df: DataFrame du corpus de textes
        :param tokenizer: tokeinzer associé au modèle prédictif
        :param labels: liste des différents labels possibles
        :return:TensorDataset du corpus
        """

        tokenizer_special_tokens = {"cls_token": tokenizer.cls_token, "cls_token_segment_id": 0,
                                    "sep_token": tokenizer.sep_token, "sep_token_extra": False,
                                    "pad_on_left": False, "cls_token_at_end": False,
                                    "pad_token": tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                    "pad_token_segment_id": 0, "pad_token_label_id": self.pad_token_label_id,
                                    "sequence_a_segment_id": 0, "mask_padding_with_zero": True}

        features = convert_to_features(corpus_df, labels, self.max_seq_length, tokenizer, tokenizer_special_tokens)

        # Convert to Tensors and build dataset
        all_text_token_ids = torch.tensor([f.text_token_ids for f in features], dtype=torch.long)
        all_text_mask = torch.tensor([f.text_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_file_token_ids = torch.tensor([f.file_token_ids for f in features], dtype=torch.long)
        all_text_parts_ids = torch.tensor([f.text_part_index for f in features], dtype=torch.long)

        dataset = TensorDataset(all_text_token_ids, all_text_mask, all_segment_ids, all_label_ids, all_file_token_ids,
                                all_text_parts_ids)
        return dataset

    def loss_with_weights(self, labels, attention_mask, preds, labels_weights):
        """
        Calcule la fonction de perte (Focal loss ou Cross Entropy loss) en prenant en compte les poids associés à chaque
        catégorie.
        :param labels: labels associés à chaque token
        :param attention_mask: masque d'attention
        :param preds: prédictions de l'algorithme pour chaque token
        :param labels_weights: poids associées à chaque classe
        :return: perte
        """

        loss = None
        if labels is not None:
            if self.loss_function == "FocalLoss":
                loss_fct = FocalLoss(alpha=labels_weights, gamma=2)
            else:
                loss_fct = CrossEntropyLoss(labels_weights)

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = preds.view(-1, len(labels_weights))
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(preds.view(-1, len(labels_weights)), labels.view(-1))

        return loss

    def parse_input_json(self):
        """
        Récupère les paramètres du fichier d'entrée au format json et les assigne à des variables de classe.
        """

        try:
            # Paramètres de configuration liés à l'entraînement (facultatifs pour de la prédiction ou de l'évaluation)
            self.adam_epsilon = self.inputs["adam_epsilon"]
            self.learning_rate = self.inputs["learning_rate"]
            self.max_steps = self.inputs["max_steps"]
            self.gradient_accumulation_steps = self.inputs["gradient_accumulation_steps"]
            self.num_train_epochs = self.inputs["num_train_epochs"]
            self.max_grad_norm = self.inputs["max_grad_norm"]
            self.warmup_steps = self.inputs["warmup_steps"]
            self.weight_decay = self.inputs["weight_decay"]
            self.loss_function = self.inputs["loss_function"]
        except (Exception,):
            _mandatory_parameters = ["adam_epsilon", "learning_rate", "max_seq_length", "max_steps",
                                     "gradient_accum, _steps", "num_train_epochs", "max_grad_norm",
                                     "per_gpu_batch_size", "warmup_steps", "weight_decay",
                                     "loss_function", "output_dir"]
            _missing_ones = [x for x in _mandatory_parameters if x not in self.inputs.keys()]
            self.logger.error(f"Missing training parameter(s): {_missing_ones}")

    def predict_with_model(self, corpus, threshold):
        """
        Détecte les entités nommées voulues dans un corpus donné.
        :param corpus: DataFrame de corpus de textes
        :param threshold: seuils de détection manuels. Si la probabilité d'une entité dépasse ce seuil, on prédit
        cette entité meme si elle ne correspond pas à la probabilité maximale.
        return: DataFrame du corpus enrichi des annotations
        """

        # Loading labels
        labels, labels_weight = self.load_labels(None)
        # Loading model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer(labels)
        # Evaluation
        predict_dataset = self.load_and_cache_texts(corpus, tokenizer, labels)

        model.to(self.device)
        _, processed_corpus = self.run_predict_and_eval(predict_dataset, model, tokenizer, labels, None,
                                                        no_evaluation=True, threshold=threshold)

        return processed_corpus

    @staticmethod
    def regex_immat(row):
        """
        Finds immats in texts using REGEX rules.
        :param row: DataFrame row
        :return: pd.Series with file, text and label columns
        """

        # Loads text
        raw_ppel = row["raw_text"]

        # REGEX immat patterns and exceptions
        regex_pattern = r"[\s\"\''\(\,\.][a-zA-Z]{2}[\s\.-]?[0-9]{3}[\s\.-]?[a-zA-Z]{2}[\s\"\''\)\,\.]"
        exceptions = ['de', 'et', 'le', 'go']

        # Finds immat patterns
        plaque = []
        for _immat in re.finditer(regex_pattern, raw_ppel):
            s = _immat.start()
            e = _immat.end()
            if not ((raw_ppel[s + 1:s + 3] in exceptions) and (raw_ppel[e - 3:e - 1] in exceptions)):
                plaque.append(raw_ppel[s + 1:e - 1])

        # Creates labels
        splitted_text = row["text"]
        if "predicted_labels" in row.keys():
            bio_tags = row["predicted_labels"]
        else:
            bio_tags = ["O" for _ in row["labels"]]
        plaque = [x.split(" ") for x in plaque]
        _ppel = splitted_text.copy()
        for _immat in plaque:
            ind = find_sub_list(_immat, _ppel)
            if ind is None:
                ind = find_sub_list(_immat, _ppel, strict=False)
            if ind is None:
                print(f"entity {_immat} not found in text")
                continue
            for i, _tag in zip(ind, _immat):
                bio_tags[i] = 'IMMAT'
            _ppel = [None for _ in _ppel[:ind[0] + len(_immat)]] + _ppel[min(len(_ppel), ind[0] + len(_immat)):]

        return pd.Series(
            {"file": row["file"], "raw_text": row["raw_text"], "text": splitted_text, "labels": row["labels"],
             "predicted_labels": bio_tags})

    @staticmethod
    def regex_email(row):
        """
        Finds e-mails in texts using REGEX rules.
        :param row: DataFrame row
        :return: pd.Series with file, text and label columns
        """

        # Loads text
        raw_ppel = row["raw_text"]

        # REGEX time patterns
        regex_pattern = r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"

        # Finds e-mail patterns
        emails = []
        for _mail in re.finditer(regex_pattern, raw_ppel):
            s = _mail.start()
            e = _mail.end()
            if raw_ppel[e - 1] == '.':
                emails.append(raw_ppel[s:e - 1])
            else:
                emails.append(raw_ppel[s:e])

        # Creates labels
        splitted_text = row["text"]
        if "predicted_labels" in row.keys():
            bio_tags = row["predicted_labels"]
        else:
            bio_tags = ["O" for _ in row["labels"]]
        emails = [x.split(" ") for x in emails]
        _ppel = splitted_text.copy()
        for _mail in emails:
            ind = find_sub_list(_mail, _ppel, strict=False)
            if ind is None:
                print(f"entity {_mail} not found in text")
                continue
            for i, _tag in zip(ind, _mail):
                bio_tags[i] = 'EMAIL'
            _ppel = [None for _ in _ppel[:ind[0] + len(_mail)]] + _ppel[min(len(_ppel), ind[0] + len(_mail)):]

        return pd.Series(
            {"file": row["file"], "raw_text": row["raw_text"], "text": splitted_text, "labels": row["labels"],
             "predicted_labels": bio_tags})

    @staticmethod
    def regex_phone(row):
        """
        Finds phone numbers in texts using REGEX rules.
        :param row: DataFrame row
        :return: pd.Series with file, text and label columns
        """

        # Loads text
        raw_ppel = row["raw_text"]
        # REGEX time patterns
        regex_pattern = [
            r"[\s\"\''\(\,\.]0[0-9][\s\.-]?([0-9]{2}[\s\.-]?){3}[0-9]{2}[\s\"\''\)\,\.]",
            r"[\s\"\''\(\,\.]\+[0-9]{1,4}[\s\.-]?[0-9][\s\.-]?([0-9]{2}[\s\.-]?){3}[0-9]{2}[\s\"\''\)\,\.]",
            r"[\s\"\''\(\,\.][0-9]{4}[\s\.-][0-9]{3}[\s\.-][0-9]{3}[\s\"\''\)\,\.]"
        ]

        # Finds phone number patterns
        phones = []
        for pattern in regex_pattern:
            for _phone in re.finditer(pattern, raw_ppel):
                s = _phone.start() + 1
                e = _phone.end() - 1
                phones.append((s, raw_ppel[s:e].strip()))
        phones.sort(key=itemgetter(0))
        phones = [x[1] for x in phones]

        # Creates labels
        splitted_text = row["text"]
        if "predicted_labels" in row.keys():
            bio_tags = row["predicted_labels"]
        else:
            bio_tags = ["O" for _ in row["labels"]]
        phones = [x.split(" ") for x in phones]
        _ppel = splitted_text.copy()
        for _phone in phones:
            ind = find_sub_list(_phone, _ppel)
            if ind is None:
                ind = find_sub_list(_phone, _ppel, strict=False)
            if ind is None:
                print(f"entity {_phone} not found in text")
                continue
            for i, _tag in zip(ind, _phone):
                bio_tags[i] = 'PHONE'
            _ppel = [None for _ in _ppel[:ind[0] + len(_phone)]] + _ppel[min(len(_ppel), ind[0] + len(_phone)):]

        return pd.Series(
            {"file": row["file"], "raw_text": row["raw_text"], "text": splitted_text, "labels": row["labels"],
             "predicted_labels": bio_tags})

    @staticmethod
    def regex_time(row):
        """
        Finds times in texts using REGEX rules.
        :param row: DataFrame row
        :return: pd.Series with file, text and label columns
        """

        # Loads text
        raw_ppel = row["raw_text"]

        # REGEX time patterns
        regex_pattern = [r"[0-9][0-9]?[\:][0-9][0-9]?", r"[0-9][0-9]?[Hh][0-9]?[0-9]?",
                         r"[0-9][0-9]?\s[hH][eE][uU][rR][eE][s]?\s[0-9]?[0-9]?",
                         r"[0-9][0-9]?\s[Hh]\s[0-9]?[0-9]?"]

        # Finds time patterns
        times = []
        for pattern in regex_pattern:
            for _time in re.finditer(pattern, raw_ppel):
                s = _time.start()
                e = _time.end()
                times.append((s, raw_ppel[s:e].strip()))
        times.sort(key=itemgetter(0))
        times = [x[1] for x in times]

        # Creates labels
        splitted_text = row["text"]
        if "predicted_labels" in row.keys():
            bio_tags = row["predicted_labels"]
        else:
            bio_tags = ["O" for _ in row["labels"]]
        times = [x.split(" ") for x in times]
        _ppel = splitted_text.copy()
        for _time in times:
            ind = find_sub_list(_time, _ppel)
            if ind is None:
                ind = find_sub_list(_time, _ppel, strict=False)
            if ind is None:
                print(f"entity {_time} not found in text")
                continue
            for i, _tag in zip(ind, _time):
                bio_tags[i] = 'TIME'
            _ppel = [None for _ in _ppel[:ind[0] + len(_time)]] + _ppel[min(len(_ppel), ind[0] + len(_time)):]

        return pd.Series(
            {"file": row["file"], "raw_text": row["raw_text"], "text": splitted_text, "labels": row["labels"],
             "predicted_labels": bio_tags})

    def run_predict_and_eval(self, dataset, model, tokenizer, labels, save_folder, no_evaluation=False, no_saving=False,
                             threshold=None):
        """

        :param dataset:
        :param model:
        :param tokenizer:
        :param labels:
        :param save_folder:
        :param no_evaluation:
        :param no_saving:
        :param threshold:
        :return:
        """

        batch_size = self.per_gpu_batch_size
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        label_map = {i: label for i, label in enumerate(labels)}
        if threshold is not None:
            threshold = {ind: threshold[ent] if ent in threshold.keys() else 1000 for ind, ent in label_map.items()}

        processed_corpus = pd.DataFrame()
        eval_loss = 0.0
        nb_eval_steps = 0
        model.to(self.device)
        model.eval()
        for batch in tqdm(dataloader, desc="Evaluating", position=0, leave=True):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                _inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3],
                           "token_type_ids": None}
                outputs = model(**_inputs)
                tmp_eval_loss, _output_probabilities = outputs[:2]
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            files, text_parts, text_tokens, real_labels, predicted_labels, labels_probabilities = \
                self.extract_info_from_batch(tokenizer, batch, _output_probabilities, label_map, threshold)

            processed_corpus = pd.concat([processed_corpus, pd.DataFrame({"file": files, "text_part": text_parts,
                                                                          "text": text_tokens,
                                                                          "labels": real_labels,
                                                                          "predicted_labels": predicted_labels,
                                                                          "labels_probabilities":
                                                                              labels_probabilities})])

        # Evaluate results
        if (not no_evaluation) & (len(processed_corpus) > 0):
            eval_loss = eval_loss / nb_eval_steps if nb_eval_steps else 0
            results = self.evaluate_and_display_results(eval_loss, processed_corpus["labels"].tolist(),
                                                        processed_corpus["predicted_labels"].tolist(),
                                                        labels, no_saving, save_folder)
        else:
            results = None

        return results, processed_corpus.reset_index(drop=True)

    def run_training(self, train_dataset, test_dataset, model, tokenizer, labels, labels_weights):
        """
        Train a transformer model.
        :param train_dataset:
        :param test_dataset:
        :param model:
        :param tokenizer:
        :param labels:
        :param labels_weights:
        :return:
        """

        train_batch_size = self.per_gpu_batch_size
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        t_total = len(train_dataloader) // self.gradient_accumulation_steps * self.num_train_epochs

        # Initializing optimizer
        optimizer, scheduler = self.set_scheduler_and_optimizer(model, t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        model.to(self.device)

        for step in range(int(self.num_train_epochs)):
            self.logger.info(f"############ EPOCH : {step + 1} / {self.num_train_epochs} ############\n")
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)
            for batch in epoch_iterator:
                # Ce n'est pas ici qu'a lieu le training, on passe simplement le modèle en mode entrainement
                model.train()
                batch = tuple(t.to(self.device) for t in batch)
                _inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3],
                           "token_type_ids": None}

                # Appelle la fonction forward de la classe RobertaForTokenClassification
                outputs = model(**_inputs)
                loss = self.loss_with_weights(batch[3], batch[1], outputs[1], labels_weights)
                loss.backward()
                tr_loss += loss.item()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                if 0 < self.max_steps < global_step:
                    epoch_iterator.close()
                    break
            # On évalue les performances du modèle à la fin de chaque epoch
            self.run_predict_and_eval(test_dataset, model, tokenizer, labels, None, no_saving=True)

        # Sauvegarge du modèle final et suppression des checkpoints
        save_path = os.path.join(self.output_dir, f"{model.__class__.__name__}.pt")
        self.save_model(save_path, model)

        return save_path

    @staticmethod
    def save_model(save_path, model):

        torch.save(model, save_path)

    def save_config_and_logs(self):

        # Export du fichier log et json
        self.output_dir = os.path.join(self.inputs["output_dir"], f"{datetime.now().strftime('%m_%d_%Y_%H%M%S')}")
        os.mkdir(self.output_dir)
        _log_file = os.path.join(self.output_dir, "log.txt")
        logging.basicConfig(filename=_log_file, level=self.log_level,
                            format='%(asctime)s %(name)s %(levelname)s:%(message)s')
        _json_file = os.path.join(self.output_dir, "config.json")
        with open(_json_file, "w") as json_file:
            json.dump(self.inputs, json_file)

    def set_scheduler_and_optimizer(self, model, t_total):

        # Linear warmup and decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps,
                                                    num_training_steps=t_total)

        return optimizer, scheduler

    @staticmethod
    def set_seed(seed_num):

        seed(seed_num)
        np.random.seed(seed_num)
        torch.manual_seed(seed_num)

    def train_model_on_corpus(self, train_corpus, test_corpus):
        """
        Entrainement d'un modèle de reconnaissance d'entités nommées.
        :param train_corpus: DataFrame du corpus de textes d'entrainement
        :param test_corpus: DataFrame du corpus de textes de test
        """

        self.parse_input_json()

        # Loading labels
        labels, labels_weight = self.load_labels(train_corpus)
        # Loading model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer(labels)
        # Loading training and eval datasets
        train_dataset = self.load_and_cache_texts(train_corpus, tokenizer, labels)
        test_dataset = self.load_and_cache_texts(test_corpus, tokenizer, labels)

        # Save config and logs
        self.save_config_and_logs()

        # Train model
        self.run_training(train_dataset, test_dataset, model, tokenizer, labels, labels_weight)

        # Show examples
        _, processed_corpus = \
            self.run_predict_and_eval(test_dataset, model, tokenizer, labels, None, no_evaluation=True)
        show_legend(self.list_entities)
        show_annotations(processed_corpus, self.list_entities, self.white_space_token)


class Pseudo:

    def __init__(self, _names_path, _address_path, _car_path, societies_path, labels_column, labels_format):

        self.names_path = _names_path
        self.address_path = _address_path
        self.car_path = _car_path
        self.societies_path = societies_path
        self.labels_col = labels_column
        self.labels_format = labels_format
        self.fake = Faker('fr_FR')
        Faker.seed()

        self.address, self.names, self.zip, self.cars, self.societies, self.train_df, self.dev_df, self.test_df = \
            [None] * 8

    def chain_of_replacements_other_entity(self, corpus, list_entities):
        """
        Remplace toutes les entités de la liste donnée par d'autres entités factices du meme type dans le corpus.
        :param corpus: DataFrame du corpus de textes
        :param list_entities: liste des entités à remplacer
        return: corpus avec entités remplacées
        """

        self.address = pd.read_csv(self.address_path)
        self.names = pd.read_csv(self.names_path)
        self.zip = self.address['postcode'].unique().tolist()
        self.cars = pd.read_csv(self.car_path)
        self.societies = pd.read_csv(self.societies_path)

        if "REF_NUM" in list_entities:
            print("REF_NUM - ", end='')
            corpus = corpus.apply(lambda x: self.replace_refnum(x), axis=1)
        if "LOC" in list_entities:
            print("LOC - ", end='')
            corpus = corpus.apply(lambda x: self.replace_loc(x), axis=1)
        if "PERSON" in list_entities:
            print("PERSON - ", end='')
            corpus = corpus.apply(lambda x: self.replace_person(x), axis=1)
        if "ORGANIZATION" in list_entities:
            print("ORGANIZATION - ", end='')
            corpus = corpus.apply(lambda x: self.replace_organization(x), axis=1)
        if "WEBSITE" in list_entities:
            print("WEBSITE - ", end='')
            corpus = corpus.apply(lambda x: self.replace_website(x), axis=1)
        if "ADDRESS" in list_entities:
            print("ADDRESS - ", end='')
            corpus = corpus.apply(lambda x: self.replace_address_zip_gpe(x), axis=1)
        if "EMAIL" in list_entities:
            print("EMAIL - ", end='')
            corpus = corpus.apply(lambda x: self.replace_email(x), axis=1)
        if "PHONE" in list_entities:
            print("PHONE - ", end='')
            corpus = corpus.apply(lambda x: self.replace_phone(x), axis=1)
        if "IMMAT" in list_entities:
            print("IMMAT - ", end='')
            corpus = corpus.apply(lambda x: self.replace_immat(x), axis=1)
        if "MONEY" in list_entities:
            print("MONEY - ", end='')
            corpus = corpus.apply(lambda x: self.replace_money(x), axis=1)
        if "DATE" in list_entities:
            print("DATE - ", end='')
            corpus = corpus.apply(lambda x: self.replace_date(x), axis=1)
        if "TIME" in list_entities:
            print("TIME - ", end='')
            corpus = corpus.apply(lambda x: self.replace_time(x), axis=1)
        if "CAR" in list_entities:
            print("CAR", end='')
            corpus = corpus.apply(lambda x: self.replace_car(x), axis=1)
        print('\n')

        return corpus

    def chain_of_replacements_tags(self, corpus, list_entities):
        """
        Remplace toutes les entités de la liste donnée par un tag au format '<Entité_n>'.
        :param corpus: DataFrame du corpus de textes
        :param list_entities: liste des entités à remplacer
        return: corpus avec entités remplacées
        """

        for entity in list_entities:
            corpus = corpus.apply(lambda x: self.replace_by_tag(x, entity), axis=1)

        return corpus

    def concat_entities(self, found_list):
        """
        Concaténation des entités ayant été partagées en plusieurs tokens successifs en une seule et meme entrée.
        :param found_list: liste des entités, chaque élément étant de type (index, label, label)
        :return: liste des entités concaténées
        """

        clean_list = []
        if found_list:
            full_entity = found_list[0][1]
            for i in range(1, len(found_list)):
                if self.labels_format == "BIO":
                    _is_same_entity = (found_list[i][0] == found_list[i - 1][0] + 1) & (found_list[i][2] == 'I')
                else:
                    _is_same_entity = (found_list[i][0] == found_list[i - 1][0] + 1)
                if _is_same_entity:
                    full_entity += ' ' + found_list[i][1]
                else:
                    clean_list.append(full_entity)
                    full_entity = found_list[i][1]
            clean_list.append(full_entity)
        else:
            clean_list = []

        return clean_list

    def create_csv_name(self, _name):
        """
        Création d'un nom propre factice à partir d'un csv.
        :param _name: nom à remplacer
        :return: nom généré
        """

        prefixes_h = ['', 'M ', 'M. ', 'Mr ', 'Mr. ', 'Monsieur ']
        weights_h = [0.1, 0.8, 0.4, 0.2, 0.1, 0.5]
        weights_h = [x / sum(weights_h) for x in weights_h]

        prefixes_f = ['', 'Mme ', 'Melle ', 'Madame ']
        weights_f = [0.1, 0.8, 0.2, 0.3]
        weights_f = [x / sum(weights_f) for x in weights_f]

        if any([x.strip() in _name.split(" ") for x in prefixes_f]):
            _prefix = np.random.choice(prefixes_f, p=weights_f)
        elif any([x.strip() in _name.split(" ") for x in prefixes_h]):
            _prefix = np.random.choice(prefixes_h, p=weights_h)
        else:
            _prefix = ""

        _firstname = np.random.choice(self.names["prenoms"].tolist(), p=self.names["prenoms_poids"].tolist())
        _lastname = np.random.choice(self.names["noms"].tolist(), p=self.names["noms_poids"].tolist()).capitalize()

        if len(_prefix) > 0:
            if random() < 0.5:
                replacement_name = f"{_prefix} {_firstname} {_lastname}"
            else:
                replacement_name = f"{_prefix} {_lastname}"
        else:
            if random() < 0.7:
                replacement_name = f"{_firstname} {_lastname}"
            else:
                replacement_name = f"{_firstname}"

        return replacement_name

    def create_faker_name(self, _name):
        """
        Création d'un nom propre factice à partir par la librairie Faker.
        :param _name: nom à remplacer
        :return: nom généré
        """

        prefixes_h = ['', 'M ', 'M. ', 'Mr ', 'Mr. ', 'Monsieur ']
        weights_h = [0.1, 0.8, 0.4, 0.2, 0.1, 0.5]
        weights_h = [x / sum(weights_h) for x in weights_h]
        male_name_generator = [self.fake.name_male, self.fake.last_name_male]
        prefixes_f = ['', 'Mme ', 'Melle ', 'Madame ']
        weights_f = [0.1, 0.8, 0.2, 0.3]
        weights_f = [x / sum(weights_f) for x in weights_f]
        female_name_generator = [self.fake.name_female, self.fake.last_name_female]

        _need_prefix = any([x.strip() in _name.split(" ") for x in prefixes_f + prefixes_h])
        if random() < 0.5:
            # male name
            if _need_prefix:
                _prefix = np.random.choice(prefixes_h, p=weights_h)
                _fake_name = np.random.choice(male_name_generator, p=[0.6, 0.4])()
            else:
                _prefix = ''
                _fake_name = np.random.choice(male_name_generator, p=[1.0, 0.0])()
            replacement_name = _prefix + _fake_name
        else:
            # female name
            if _need_prefix:
                _prefix = np.random.choice(prefixes_f, p=weights_f)
                _fake_name = np.random.choice(female_name_generator, p=[0.6, 0.4])()
            else:
                _prefix = ''
                _fake_name = np.random.choice(female_name_generator, p=[1.0, 0.0])()
            replacement_name = _prefix + _fake_name

        return replacement_name

    @staticmethod
    def match_case(new_entity, old_entity):
        """
        Fait correspondre le style de l'entité de remplacement avec celui de l'entité originale afin de respecter la
        casse.
        :param new_entity: entité de remplacement
        :param old_entity: entité originale
        :return: entité de remplacement formatée
        """
        entity_letters = [x for x in old_entity if x not in ' ' + string.digits]
        if len(entity_letters) == 0:
            return new_entity
        # Make the fake name lower or upper depending on the text style
        if sum(1 for c in entity_letters if c.isupper()) / len(entity_letters) > 0.8:
            new_entity = new_entity.upper()
        if sum(1 for c in entity_letters if c.islower()) / len(entity_letters) > 0.999:
            new_entity = new_entity.lower()
        return new_entity

    @staticmethod
    def match_ponctuation(new_entity, old_entity):
        """
        Ajoute un point ou une virgule à la fin de l'entité s'il y en avait un à l'origine.
        :param new_entity: entité de remplacement
        :param old_entity: entité originale
        :return: entité de remplacement formatée
        """

        # looking for possible endings to reproduce
        endings = ['.<br>', ',<br>', ',', '.']
        for _ending in endings:
            if old_entity[-len(_ending):] == _ending:
                new_entity += _ending
        return new_entity

    @staticmethod
    def oversample_corpus(corpus, test_corpus, generated_ppel, list_ppel, distinct_ppel=0, method="first",
                          entities=None, ratio=0.5):
        """
        Loads all ppel .txt and .ann files and stores information into a dataframe.
        :param corpus:
        :param test_corpus:
        :param generated_ppel: number of artificial ppel to generate
        :param list_ppel:
        :param distinct_ppel: number of ppel used for generating new ones.
        :param method:
        :param entities:
        :param ratio:
        """
        if entities is None:
            entities = []
        _error_message = "The parameter 'method' must be equal to 'first' (default value), 'random' or 'balanced'"
        assert method in ["first", "random", "balanced"], _error_message
        if not list_ppel:
            if distinct_ppel == 0:
                corpus = corpus.sample(generated_ppel, replace=True)
            else:
                assert distinct_ppel <= len(
                    corpus), "The number of texts must be smaller or equal to the corpus length"
                if method == "first":
                    test_corpus = corpus.iloc[distinct_ppel:].copy()
                    # corpus = corpus.iloc[:distinct_ppel].sample(generated_ppel, replace=True)
                    corpus = corpus.iloc[:distinct_ppel]
                    corpus = corpus[corpus["label"].apply(lambda x: "PERSON" in " ".join(x))].sample(
                        generated_ppel, replace=True)

                elif method == "random":
                    test_corpus = corpus.sample(len(corpus) - distinct_ppel)
                    corpus = corpus[~corpus["file"].isin(test_corpus["file"].unique())].\
                        sample(generated_ppel, replace=True)
                elif method == "balanced":
                    assert 0 < ratio < 1, "The ratio must be in ]0, 1[."
                    assert len(entities) > 0, "If the method 'balanced' is selected, you must define the entities."
                    # test_corpus = corpus.sample(len(corpus) - distinct_ppel)
                    # _train_corpus = corpus[~corpus["file"].isin(test_corpus["file"].unique())]
                    _train_corpus = corpus.drop_duplicates(["file"])
                    _train_positive = _train_corpus[
                        _train_corpus["label"].apply(lambda x: any([y.split("-")[-1] in entities for y in x]))]
                    _train_negative = _train_corpus[
                        _train_corpus["label"].apply(lambda x: not any([y.split("-")[-1] in entities for y in x]))]
                    n_pos = int(ratio * generated_ppel)
                    corpus = pd.concat([_train_positive.sample(n_pos, replace=True),
                                        _train_negative.sample(generated_ppel - n_pos, replace=True)], axis=0).sample(
                        frac=1)
        else:
            corpus = corpus[corpus.file in list_ppel].sample(generated_ppel, replace=True)

        return corpus, test_corpus

    def replace_by_tag(self, x, entity):
        """
        Remplace une entité détectée par un tag au format '<Entité_n>'.
        :param x: ligne du DataFrame à traiter
        :param entity: entité à remplacer
        :return: ligne modifiée
        """

        x["text_anonyme"] = x["text"].copy()
        entities = [(ind, x["text"][ind], entity.split('-')[0]) for ind, ent in enumerate(x["labels"]) if
                    ent.split('-')[-1] == entity]
        clean_entities = self.concat_entities(entities)
        replacement_tags = {y: f"<{entity}_{i + 1}>" for i, y in enumerate(list(dict.fromkeys(clean_entities)))}

        for _ent in clean_entities:
            x["text_anonyme"], x["labels"] = switch_entity(x, _ent, replacement_tags[_ent], entity, split_first=False)

        return x

    def replace_person(self, x, method="dataset"):
        """
        Remplace les noms propres d'un texte par d'autres noms factices.
        :param x: ligne du DataFrame à traiter
        :param method: méthode de remplacement (dataset ou Faker)
        :return: ligne modifiée
        """
        person_names = [(ind, x.text[ind], entity.split('-')[0]) for ind, entity in enumerate(x[self.labels_col]) if
                        entity.split('-')[-1] == "PERSON"]
        clean_names = self.concat_entities(person_names)

        pers_dic = {}
        for _name in clean_names:
            replacement_name = None
            # If we already met this entity, we use the same substitution again
            if _name in pers_dic.keys():
                replacement_name = pers_dic[_name]
            # If this is the first time we meet this entity, we create a substitution
            else:
                # Creating a fake replacement name
                if method == "dataset":
                    replacement_name = self.create_csv_name(_name)
                if method == "Faker":
                    replacement_name = self.create_faker_name(_name)
                replacement_name = self.match_case(replacement_name, _name)
                pers_dic[_name] = replacement_name
                replacement_name = self.match_ponctuation(replacement_name, _name)
            x.text, x[self.labels_col] = self.switch_entity(x, _name, replacement_name, 'PERSON')

        return x

    def replace_organization(self, x):
        """
        Remplace les sociétés d'un texte par d'autres sociétés factices.
        :param x: ligne du DataFrame à traiter
        :return: ligne modifiée
        """
        gafam = ['Google', 'Amazon', 'Facebook', 'Apple', 'Microsoft']
        banks = ['BNP PARIBAS', 'Boursorama', 'Crédit agricole', "Caisse d'épargne", "Société générale",
                 'Crédit Mutuel', 'Banque populaire', 'Banque postale']
        malls = ['Agora', 'Atac', 'Auchan', 'Carrefour', 'Carrefour Market', 'Casino', 'Coop', 'Cora', 'Costco',
                 'Douka Be',
                 'E. Leclerc', 'Entrepot Produits Frais', 'Intermarche', 'Leader Price', 'Leclerc', 'Simply Market',
                 'Spar', 'Super U',
                 'Supermarche Match']
        security = ['Gendarmerie', 'Commissariat', 'Police municipale', 'Police']
        internet = ['OVH', 'Gandhi', '1&1', 'GoDaddy']
        telecom = ['Orange', 'SFR', 'Bouygues Telecom', 'Free', 'Sosh']
        prefixes = ['société', 'sté', 'SARL', 'sci']

        all_lists = [gafam, banks, malls, security, internet, telecom]
        found_org = [(ind, x.text[ind], entity.split('-')[0]) for ind, entity in enumerate(x[self.labels_col]) if
                     entity.split('-')[-1] == "ORGANIZATION"]
        clean_org = self.concat_entities(found_org)

        org_dic = {}
        if len(clean_org) > 0:
            replacement_org = None
            for org in clean_org:
                # If we already met this entity, we use the same substitution again
                if org in org_dic.keys():
                    replacement_org = org_dic[org]
                # If this is the first time we meet this entity, we create a substitution
                else:
                    # Remplacement du préfixe s'il existe
                    _prefix = [x for x in prefixes if x in org.split(" ")]
                    if len(_prefix) > 0:
                        _prefix = choice(prefixes)
                    else:
                        _prefix = ""
                    # Remplacement du nom de la société
                    _found_something = False
                    # S'il s'agit d'une société fréquemment citée, on la remplace par une autre du meme type par soucis
                    # de cohérence
                    for list_org in all_lists:
                        _l_splitted = [[unidecode.unidecode(y).lower() for y in x.split(' ')] for x in list_org]
                        _o_splitted = [unidecode.unidecode(y).lower() for y in org.split(' ')]
                        known_pattern = set(
                            list_org[ind] for ind, x in enumerate(_l_splitted) if
                            find_sub_list(x, _o_splitted) is not None)
                        if len(known_pattern) > 0:
                            _found_something = True
                            org_dic[list(known_pattern)[0]] = choice(list_org)
                            replacement_org = org_dic[list(known_pattern)[0]]
                        break
                    # S'il s'agit d'une société moins connue, on la remplace par une société sélectionnée aléatoirement
                    # dans la base Infogreffe
                    if not _found_something:
                        replacement_org = self.societies.sample(1).iloc[0]["Dénomination"].strip().capitalize()
                    replacement_org = _prefix + " " + replacement_org
                    replacement_org = self.match_case(replacement_org, org).strip()
                    org_dic[org] = replacement_org
                    replacement_org = self.match_ponctuation(replacement_org, org)
                # Modification du texte
                x.text, x[self.labels_col] = self.switch_entity(x, org, replacement_org, 'ORGANIZATION')
        return x

    def replace_loc(self, x):
        """
        Remplace les emplacements d'un texte par d'autres emplacements factices.
        :param x: ligne du DataFrame à traiter
        :return: ligne modifiée
        """

        _places = ["de la place", "du supermarché", "du centre commercial", "de l'hopital", "du cinéma", "de la mairie",
                   "du centre", "du marché", "de la bilbiothèque", "du magasin", "du parc"]
        found_loc = [(ind, x.text[ind], entity.split('-')[0]) for ind, entity in enumerate(x[self.labels_col]) if
                     entity.split('-')[-1] == "LOC"]
        clean_loc = self.concat_entities(found_loc)

        for _loc in clean_loc:
            if 'parking' in unidecode.unidecode(_loc).lower():
                _city = self.address['city'].sample(1).iloc[0]
                _place = choice(_places)
                replacement_loc = choice([f"parking {_place} de {_city}", f"parking de {_city}"])
            elif 'gare' in unidecode.unidecode(_loc).lower():
                _city = self.address['city'].sample(1).iloc[0]
                replacement_loc = f"gare de {_city}"
            elif 'parc' in unidecode.unidecode(_loc).lower():
                _city = self.address['city'].sample(1).iloc[0]
                replacement_loc = f"parc de {_city}"
            elif 'gendarmerie' in unidecode.unidecode(_loc).lower():
                _city = self.address['city'].sample(1).iloc[0]
                replacement_loc = f"gendarmerie de {_city}"
            elif 'commissariat' in unidecode.unidecode(_loc).lower():
                _city = self.address['city'].sample(1).iloc[0]
                replacement_loc = f"commissariat de {_city}"
            elif ('rond-point' in unidecode.unidecode(_loc).lower()) or (
                    'rond point' in unidecode.unidecode(_loc).lower()):
                _city = self.address['city'].sample(1).iloc[0]
                _place = choice(_places)
                replacement_loc = choice([f"rond-point {_place} de {_city}", f"rond-point de {_city}",
                                          f"rond point {_place} de {_city}", f"rond point de {_city}"])
            elif ('station service' in unidecode.unidecode(_loc).lower()) or (
                    'station essence' in unidecode.unidecode(_loc).lower()):
                _city = self.address['city'].sample(1).iloc[0]
                _place = choice(_places)
                replacement_loc = choice([f"station service {_place} de {_city}", f"station service de {_city}",
                                          f"station essence {_place} de {_city}", f"station essence de {_city}"])
            elif 'restaurant' in unidecode.unidecode(_loc).lower():
                _nom = choice([self.fake.last_name_male, self.fake.last_name_female])()
                replacement_loc = choice([f"restaurant Chez {_nom}", f"restaurant le {self.fake.word()}"])
            elif 'hotel' in unidecode.unidecode(_loc).lower():
                replacement_loc = f"restaurant le {self.fake.word()}"
            elif 'residence' in unidecode.unidecode(_loc).lower():
                _nom = choice([self.fake.last_name_male, self.fake.last_name_female])()
                replacement_loc = choice([f"résidence Le {_nom}"])
            else:
                replacement_loc = _loc

            replacement_loc = self.match_case(replacement_loc, _loc)
            replacement_loc = self.match_ponctuation(replacement_loc, _loc)
            x.text, x[self.labels_col] = self.switch_entity(x, _loc, replacement_loc, 'LOC')

        return x

    def replace_date(self, x):
        """
        Remplace les dates d'un texte par d'autres dates factices.
        :param x: ligne du DataFrame à traiter
        :return: ligne modifiée
        """

        locale.setlocale(locale.LC_ALL, 'fr_FR.utf-8')
        # date_formats_with_days = ["%d/%m/%Y", "%d/%m/%y", "%d %b %Y", "%d-%m-%Y", "%d-%m-%y", "%d.%m.%Y", "%d.%m.%y",
        #                           "%d %b", "%A %d/%m/%Y", "%A %d/%m/%y", "%A %d %b %Y", "%A %d %b", "%-d/%-m/%Y",
        #                           "%-d/%-m/%y", "%-d %b %Y", "%-d-%-m-%Y", "%-d-%-m-%y", "%-d.%-m.%Y", "%-d.%-m.%y",
        #                           "%-d %b", "%A %-d/%-m/%Y", "%A %-d/%-m/%y", "%A %-d %b %Y", "%A %-d %b"]
        date_formats_without_days = ["%d/%m/%Y", "%d/%m/%y", "%d %b %Y", "%d-%m-%Y", "%d-%m-%y", "%d.%m.%Y", "%d.%m.%y",
                                     "%d %b", "%-d/%-m/%Y", "%-d/%-m/%y", "%-d %b %Y", "%-d-%-m-%Y", "%-d-%-m-%y",
                                     "%-d.%-m.%Y", "%-d.%-m.%y", "%-d %b"]
        weights = [1, 1, 0.8, 0.3, 0.3, 0.1, 0.1, 0.2, 1, 1, 0.8, 0.3, 0.3, 0.1, 0.1, 0.2]
        weights = [x / sum(weights) for x in weights]
        month_list = ['janvier', 'fevrier', 'mars', 'avril', 'mai', 'juin', 'juillet', 'aout', 'septembre', 'octobre',
                      'novembre', 'decembre']

        found_dates = [(ind, x.text[ind], entity.split('-')[0]) for ind, entity in enumerate(x[self.labels_col]) if
                       entity.split('-')[-1] == "DATE"]
        clean_dates = self.concat_entities(found_dates)

        if len(found_dates) > 0:
            for _date in clean_dates:
                # month only
                if unidecode.unidecode(_date.split(' ')[0]) in month_list:
                    replacement_date = datetime.now() - timedelta(days=randint(0, 365))
                    replacement_date = replacement_date.strftime("%b %Y")
                    replacement_date = self.match_case(replacement_date, _date)
                    replacement_date = self.match_ponctuation(replacement_date, _date)
                else:
                    replacement_date = datetime.now() - timedelta(days=randint(0, 365))
                    replacement_date = replacement_date.strftime(np.random.choice(date_formats_without_days, p=weights))
                    replacement_date = self.match_case(replacement_date, _date)
                    replacement_date = self.match_ponctuation(replacement_date, _date)
                x.text, x[self.labels_col] = self.switch_entity(x, _date, replacement_date, 'DATE')

        return x

    def replace_time(self, x):
        """
        Remplace les heures d'un texte par d'autres heures factices.
        :param x: ligne du DataFrame à traiter
        :return: ligne modifiée
        """

        locale.setlocale(locale.LC_ALL, 'fr_FR.utf-8')
        time_formats = ["%Hh%M", "%HH%M", "%-Hh%-M", "%-HH%-M", "%H:%M", "%H heures %M", "%-H heures %-M", "%H h %M",
                        "%H H %M",
                        "%-H h %-M", "%-H H %-M"]
        weights = [1, 0.1, 0.1, 0.05, 0.05, 0.3, 0.05, 0.02, 0.02, 0.02, 0.02]
        weights = [x / sum(weights) for x in weights]

        found_times = [(ind, x.text[ind], entity.split('-')[0]) for ind, entity in enumerate(x[self.labels_col]) if
                       entity.split('-')[-1] == "TIME"]
        clean_times = self.concat_entities(found_times)

        if len(clean_times) > 0:
            former_text = ' '.join(x.text)
            free_format = [True for _ in clean_times]
            if len(clean_times) > 1:
                for i in range(len(clean_times) - 1):
                    _link = former_text[former_text.index(clean_times[i]) + len(clean_times[i]):former_text.index(
                        clean_times[i + 1])]
                    _link = unidecode.unidecode(_link).strip()
                    if (_link == 'et') or (_link == 'a'):
                        free_format[i + 1] = False
            _previous_time, _previous_format = [None] * 2
            for ind, _time in enumerate(clean_times):
                if free_format[ind]:
                    replacement_time = datetime.now().replace(hour=randint(0, 23), minute=randint(0, 59))
                    _previous_time = replacement_time
                    _previous_format = np.random.choice(time_formats, p=weights)
                else:
                    replacement_time = _previous_time + timedelta(minutes=randint(0, 180))
                replacement_time = replacement_time.strftime(_previous_format)
                replacement_time = self.match_case(replacement_time, _time)
                replacement_time = self.match_ponctuation(replacement_time, _time)
                x.text, x[self.labels_col] = self.switch_entity(x, _time, replacement_time, 'TIME')

        return x

    def replace_address_zip_gpe(self, x):
        """
        Remplace les adresses, code postaux et entités géopolitiques d'un texte par d'autres factices.
        :param x: ligne du DataFrame à traiter
        :return: ligne modifiée
        """

        found_zip = [(ind, x.text[ind], entity.split('-')[0]) for ind, entity in enumerate(x[self.labels_col]) if
                     entity.split('-')[-1] == "ZIP"]
        found_addresses = [(ind, x.text[ind], entity.split('-')[0]) for ind, entity in enumerate(x[self.labels_col]) if
                           entity.split('-')[-1] == "ADDRESS"]
        found_gpe = [(ind, x.text[ind], entity.split('-')[0]) for ind, entity in enumerate(x[self.labels_col]) if
                     entity.split('-')[-1] == "GPE"]

        # If there is at least one person name, replace a random one. Else return None.
        clean_addresses = self.concat_entities(found_addresses)
        clean_zip = self.concat_entities(found_zip)
        clean_zip = [str(x) for x in clean_zip]
        clean_gpe = self.concat_entities(found_gpe)

        for _address in clean_addresses:
            with_num = sum(c.isdigit() for c in _address)
            if with_num > 0:
                _random_place = self.address.sample(1).iloc[0]
                replacement_address = "{} {}".format(randint(1, 60), _random_place.street)
            else:
                ind = min(find_sub_list(_address.split(' '), x.text)) - 1
                word_before = x.text[ind] if ind > 0 else None
                if word_before == 'la':
                    _compat_types = ['Résidence', 'Rue', 'Cite', 'Place', 'Route', 'Residence', 'Ruelle', 'Traverse',
                                     'Ferme', 'Voie']
                    compat_df = self.address[self.address['street_type'].isin(_compat_types)]
                elif word_before == 'le':
                    _compat_types = ['Lieu', 'Chemin', 'Lotissement', 'Square', 'Sentier', 'Domaine', 'Hameau', 'Clos',
                                     'Quai', 'Chez', 'Passage', 'Boulevard', 'Lieu-dit', 'Cour']
                    compat_df = self.address[self.address['street_type'].isin(_compat_types)]
                elif word_before == 'l':
                    _compat_types = ['Allée', 'Impasse', 'Avenue', 'Allee']
                    compat_df = self.address[self.address['street_type'].isin(_compat_types)]
                else:
                    compat_df = self.address

                _random_place = compat_df.sample(1).iloc[0]
                replacement_address = "{}".format(_random_place.street)

            replacement_zip = str(_random_place.postcode)
            replacement_gpe = _random_place.city
            replacement_address = self.match_case(replacement_address, _address)
            replacement_address = self.match_ponctuation(replacement_address, _address)
            x.text, x[self.labels_col] = self.switch_entity(x, _address, replacement_address, 'ADDRESS')

            if len(clean_zip):
                x.text, x[self.labels_col] = self.switch_entity(x, clean_zip[0], replacement_zip, 'ZIP')
                del clean_zip[0]
            if len(clean_gpe):
                replacement_gpe = self.match_case(replacement_gpe, clean_gpe[0])
                replacement_gpe = self.match_ponctuation(replacement_gpe, clean_gpe[0])
                x.text, x[self.labels_col] = self.switch_entity(x, clean_gpe[0], replacement_gpe, 'GPE')
                del clean_gpe[0]

        for _zip in clean_zip:
            _random_place = self.address.sample(1).iloc[0]
            replacement_zip = str(_random_place.postcode)
            replacement_gpe = _random_place.city
            x.text, x[self.labels_col] = self.switch_entity(x, _zip, replacement_zip, 'ZIP')

            if len(clean_gpe):
                replacement_gpe = self.match_case(replacement_gpe, clean_gpe[0])
                replacement_gpe = self.match_ponctuation(replacement_gpe, clean_gpe[0])
                x.text, x[self.labels_col] = self.switch_entity(x, clean_gpe[0], replacement_gpe, 'GPE')
                del clean_gpe[0]

        gpe_dict = {}
        for _gpe in clean_gpe:
            # If we already met this entity, we use the same substitution again
            if _gpe in gpe_dict.keys():
                replacement_gpe = gpe_dict[_gpe]
            # If this is the first time we meet this entity, we create a substitution
            else:
                _random_place = self.address.sample(1).iloc[0]
                replacement_gpe = _random_place.city
                replacement_gpe = self.match_case(replacement_gpe, _gpe)
                replacement_gpe = self.match_ponctuation(replacement_gpe, _gpe)
                gpe_dict[_gpe] = replacement_gpe
            x.text, x[self.labels_col] = self.switch_entity(x, _gpe, replacement_gpe, 'GPE')

        return x

    def replace_email(self, x):
        """
        Remplace les e-mails d'un texte par d'autres e-mails factices.
        :param x: ligne du DataFrame à traiter
        :return: ligne modifiée
        """

        generators = [self.fake.ascii_company_email, self.fake.ascii_email, self.fake.ascii_free_email,
                      self.fake.ascii_safe_email]
        weights = [0.4, 0.5, 1, 0.1]
        weights = [x / sum(weights) for x in weights]

        found_mails = [(ind, x.text[ind], entity.split('-')[0]) for ind, entity in enumerate(x[self.labels_col]) if
                       entity.split('-')[-1] == "EMAIL"]
        clean_emails = self.concat_entities(found_mails)

        mail_dic = {}
        for _mail in clean_emails:
            # If we already met this entity, we use the same substitution again
            if _mail in mail_dic.keys():
                replacement_mail = mail_dic[_mail]
            # If this is the first time we meet this entity, we create a substitution
            else:
                replacement_mail = np.random.choice(generators, p=weights)()
                replacement_mail = self.match_case(replacement_mail, _mail)
                replacement_mail = self.match_ponctuation(replacement_mail, _mail)
                mail_dic[_mail] = replacement_mail
            x.text, x[self.labels_col] = self.switch_entity(x, _mail, replacement_mail, 'EMAIL')

        return x

    def replace_phone(self, x):
        """
        Remplace les numéros de téléphone d'un texte par d'autres numéros factices.
        :param x: ligne du DataFrame à traiter
        :return: ligne modifiée
        """

        found_phones = [(ind, x.text[ind], entity.split('-')[0]) for ind, entity in enumerate(x[self.labels_col]) if
                        entity.split('-')[-1] == "PHONE"]
        clean_phones = self.concat_entities(found_phones)

        for _phone in clean_phones:
            replacement_phone = self.fake.phone_number()
            replacement_phone = self.match_case(replacement_phone, _phone)
            replacement_phone = self.match_ponctuation(replacement_phone, _phone)
            x.text, x[self.labels_col] = self.switch_entity(x, _phone, replacement_phone, 'PHONE')

        return x

    def replace_immat(self, x):
        """
        Remplace les immatriculations d'un texte par d'autres immatriculations factices.
        :param x: ligne du DataFrame à traiter
        :return: ligne modifiée
        """

        links = ['-', '.', '', ' ']
        weights = [1, 0.1, 0.1, 0.1]
        weights = [x / sum(weights) for x in weights]

        found_immat = [(ind, x.text[ind], entity.split('-')[0]) for ind, entity in enumerate(x[self.labels_col]) if
                       entity.split('-')[-1] == "IMMAT"]
        clean_immat = self.concat_entities(found_immat)

        for _immat in clean_immat:
            _link = np.random.choice(links, p=weights)
            replacement_immat = [choice(string.ascii_uppercase) for _ in range(2)] + [_link] + [choice(string.digits)
                                                                                                for _ in range(3)] + [
                                    _link] + [choice(string.ascii_uppercase) for _ in range(2)]
            replacement_immat = ''.join(replacement_immat)
            x.text, x[self.labels_col] = self.switch_entity(x, _immat, replacement_immat, 'IMMAT')

        return x

    def replace_money(self, x):
        """
        Remplace les sommes d'argent d'un texte par d'autres sommes factices.
        :param x: ligne du DataFrame à traiter
        :return: ligne modifiée
        """

        currency = [' €', '€', ' euros', 'euros', ' Euros']
        weights = [1, 0.5, 0.5, 0.01, 0.02]
        weights = [x / sum(weights) for x in weights]

        found_money = [(ind, x.text[ind], entity.split('-')[0]) for ind, entity in enumerate(x[self.labels_col]) if
                       entity.split('-')[-1] == "MONEY"]
        clean_money = self.concat_entities(found_money)

        for _money in clean_money:
            alpha = random()
            if alpha < 0.5:
                _amount = "{0:.2f}".format(abs(np.random.normal(20, 100)))
            else:
                _amount = str(int(abs(np.random.normal(20, 100))))
            _currency = np.random.choice(currency, p=weights)
            replacement_money = _amount + np.random.choice(currency, p=weights)
            replacement_money = ''.join(replacement_money)
            replacement_money = self.match_ponctuation(replacement_money, _money)
            x.text, x[self.labels_col] = self.switch_entity(x, _money, replacement_money, 'MONEY')

        return x

    def replace_website(self, x):
        """
        Remplace les sites internet d'un texte par d'autres sites factices.
        :param x: ligne du DataFrame à traiter
        :return: ligne modifiée
        """

        websites = ['Amazon', 'Facebook', 'Google', 'OVH', 'Orange', 'Paypal', 'Le bon coin', "Bon Coin", "Boncoin",
                    'Leboncoin', 'Microsoft', 'Instagram', 'Booking.com', 'Airbnb']
        domain_name = r"(\.[a-zA-Z]{2,3}\Z)"
        weights = [1 for _ in websites]
        weights = [x / sum(weights) for x in weights]

        found_website = [(ind, x.text[ind], entity.split('-')[0]) for ind, entity in enumerate(x[self.labels_col]) if
                         entity.split('-')[-1] == "WEBSITE"]
        clean_website = self.concat_entities(found_website)

        web_dict = {}
        for _website in clean_website:
            # If we already met this entity, we use the same substitution again
            if _website in web_dict.keys():
                replacement_web = web_dict[_website]
            # If this is the first time we meet this entity, we create a substitution
            else:
                if (_website.lower().startswith("http")) or (_website.lower().startswith("http")) or bool(
                        re.search(domain_name, _website)):
                    replacement_web = self.fake.url()
                    if random() < 0.9:
                        replacement_web = replacement_web.replace('https://', '').replace('http://', '').replace('/',
                                                                                                                 '')
                    if random() < 0.7:
                        replacement_web = replacement_web.replace('www.', '')
                else:
                    replacement_web = np.random.choice(websites, p=weights)
                replacement_web = self.match_case(replacement_web, _website)
                replacement_web = self.match_ponctuation(replacement_web, _website)
                web_dict[_website] = replacement_web
            x.text, x[self.labels_col] = self.switch_entity(x, _website, replacement_web, 'WEBSITE')

        return x

    def replace_car(self, x):
        """
        Remplace les véhicules d'un texte par d'autres véhicules factices.
        :param x: ligne du DataFrame à traiter
        :return: ligne modifiée
        """

        colors_f = ['grise', 'blanche', 'noire', 'rouge', 'bleue', 'verte', 'beige']
        colors_h = ['gris', 'blanc', 'noir', 'rouge', 'bleu', 'vert', 'beige']

        found_car = [(ind, x.text[ind], entity.split('-')[0]) for ind, entity in enumerate(x[self.labels_col]) if
                     entity.split('-')[-1] == "CAR"]
        clean_car = self.concat_entities(found_car)

        for _car in clean_car:
            if np.any([x in unidecode.unidecode(_car).lower() for x in ['voiture', 'vehicule']]):
                if 'vehicule' in unidecode.unidecode(_car).lower():
                    all_cars = self.cars[self.cars['genre'] == 'H']
                    random_car = all_cars.sample(1).iloc[0]
                    _rcm = random_car['marque']
                    _rct = random_car['modele']
                    colf = choice(colors_f)
                    colh = choice(colors_h)
                    formats = [f"véhicule {_rcm} {_rct}", f"véhicule {_rcm} {_rct} {colh}",
                               f"véhicule {_rcm} {_rct} de couleur {colf}",
                               f"véhicule ({_rcm} {_rct})", f"{_rcm} {_rct}", f"{_rct}"]
                else:
                    all_cars = self.cars[self.cars['genre'] == 'F']
                    random_car = all_cars.sample(1).iloc[0]
                    _rcm = random_car['marque']
                    _rct = random_car['modele']
                    colf = choice(colors_f)
                    formats = [f"voiture {_rcm} {_rct} {colf}", f"voiture {_rcm} {_rct}", f"{_rcm} {_rct}", f"{_rct}"]
            else:
                ind = None
                try:
                    ind = find_sub_list(_car.split(' '), x.text)[0] - 1
                except (Exception, ):
                    print(f"entity of type {_car} not found")
                if x.text[ind] in ['un', 'mon', 'son', 'le']:
                    all_cars = self.cars[self.cars['genre'] == 'H']
                    random_car = all_cars.sample(1).iloc[0]
                    _rcm = random_car['marque']
                    _rct = random_car['modele']
                    colf = choice(colors_f)
                    colh = choice(colors_h)
                    formats = [f"véhicule {_rcm} {_rct}", f"véhicule {_rcm} {_rct} {colh}",
                               f"véhicule {_rcm} {_rct} de couleur {colf}",
                               f"véhicule ({_rcm} {_rct})", f"{_rcm} {_rct}", f"{_rct}"]
                else:
                    all_cars = self.cars[self.cars['genre'] == 'F']
                    random_car = all_cars.sample(1).iloc[0]
                    _rcm = random_car['marque']
                    _rct = random_car['modele']
                    colf = choice(colors_f)
                    formats = [f"voiture {_rcm} {_rct} {colf}", f"voiture {_rcm} {_rct}", f"{_rcm} {_rct}", f"{_rct}"]

            replacement_car = choice(formats)
            replacement_car = self.match_case(replacement_car, _car)
            replacement_car = self.match_ponctuation(replacement_car, _car)
            x.text, x[self.labels_col] = self.switch_entity(x, _car, replacement_car, 'CAR')

        return x

    def replace_refnum(self, x):
        """
        Remplace les numéros de référence d'un texte par d'autres numéros factices.
        :param x: ligne du DataFrame à traiter
        :return: ligne modifiée
        """

        categories = ['rapport', 'contravention', 'constatation', 'commande', 'facture', 'courante', 'numero', 'iban',
                      'reference', 'bancaire', 'verbal', 'adresse', 'contrat', 'suivi', 'carte', 'cheque', 'code',
                      'colis', 'compte', 'dossier', 'puce', 'siret', 'siren', 'proces', 'ipv6', 'avis',
                      'produit', 'serie', 'bic', 'imei', 'ip', 'main', 'n°', 'ref', 'de', 'vol', ':']
        found_ref = [(ind, x.text[ind], entity.split('-')[0]) for ind, entity in enumerate(x[self.labels_col]) if
                     entity.split('-')[-1] == "REF_NUM"]
        clean_ref = self.concat_entities(found_ref)

        for _ref in clean_ref:
            ref_print = _ref
            clean_ref = _ref
            for cat in categories:
                ind = '°'.join([unidecode.unidecode(x).lower() for x in clean_ref.split('°')]).find(cat)
                if ind != -1:
                    clean_ref = clean_ref[:ind] + clean_ref[ind + len(cat):]
                    ind_or = '°'.join([unidecode.unidecode(x).lower() for x in _ref.split('°')]).find(cat)
                    ref_print = ref_print[:ind_or] + ''.join([' ' for _ in cat]) + ref_print[ind_or + len(cat):]
            clean_ref = clean_ref.strip()
            _temp = []
            for _char in clean_ref:
                if _char in string.ascii_lowercase:
                    _temp.append(choice(string.ascii_lowercase))
                elif _char in string.ascii_uppercase:
                    _temp.append(choice(string.ascii_uppercase))
                elif _char in string.digits:
                    _temp.append(choice(string.digits))
                elif _char in [' ', '-', '(', ')', '.', ':']:
                    _temp.append(_char)
                else:
                    _temp.append(choice(string.punctuation))
            _temp = ''.join(_temp)

            ind = _ref.find(clean_ref)
            replacement_ref = _ref[:ind] + _temp + _ref[ind + len(_temp):]

            replacement_ref = self.match_case(replacement_ref, _ref)
            replacement_ref = self.match_ponctuation(replacement_ref, _ref)
            x.text, x[self.labels_col] = self.switch_entity(x, _ref, replacement_ref, 'REF_NUM')

        return x

    def switch_entity(self, x, old_entity, new_entity, entity_type):
        """
        Remplace une entité par une autre dans un texte.
        :param x: ligne du DataFrame à traiter
        :param old_entity: entité à remplacer
        :param new_entity: nouvelle entité à insérer
        :param entity_type: type de l'entité à remplacer (sans B- ou I-)
        :return: texte et labels modifiés
        """

        def _find_sub_list(sl, ll, start=0):
            """
            Finds a sublist in another list and returns the starting and ending indices.
            :param sl: sublist
            :param ll: list
            :param start : starting index (all elements before that will be ignored by the search)
            :return: starting and ending positions of the sublist in the list
            """
            sll = len(sl)
            ll = [None for _ in range(start)] + ll[start:]
            for ind in (i for i, e in enumerate(ll) if e == sl[0]):
                if ll[ind:ind + sll] == sl:
                    return ind, ind + sll - 1
            return None, None

        found_right_entity = False
        char_start = 0
        max_pos = len(x[self.labels_col]) - [x.split('-')[-1] for x in x[self.labels_col]][::-1].index(entity_type) - 1
        i_s, i_e = [None] * 2
        while not found_right_entity:
            i_s, i_e = _find_sub_list(old_entity.split(' '), x.text, char_start)
            found_label = None if x[self.labels_col][i_s] == 'O' else x[self.labels_col][i_s].split("-")[-1]

            # we check whether we could find the entity in the text and if the corresponding labels are the right ones.
            # This prevents replacing a word or a sequence contained inside other entities.
            if (i_s is not None) & (found_label == entity_type):
                found_right_entity = True
            else:
                char_start = i_e + 1
                if char_start > max_pos:
                    break
        assert found_right_entity, "the identity we want to replace was not found"

        x.text = x.text[:i_s] + new_entity.split(' ') + x.text[i_e + 1:]
        entity_labels = ['I-' + entity_type for _ in new_entity.split(' ')]
        if len(entity_labels) > 1:
            entity_labels[0] = 'B-' + entity_type
        x[self.labels_col] = x[self.labels_col][:i_s] + entity_labels + x[self.labels_col][i_e + 1:]

        return x.text, x[self.labels_col]


def detect_entities(_inputs, corpus, threshold=None):
    """
    Détecte les entités nommées sélectionnées dans le corpus donné en argument.
    :param _inputs: paramètres d'entrainement du modèle
    :param corpus: corpus à annoter
    :param threshold: seuils de détection manuels. Si la probabilité d'une catégorie dépasse ce seuil, on prédit cette
    catégorie meme si elle ne correspond pas à la probabilité maximale.
    :return: corpus avec prédictions sur la nature des entités
    """
    # Initialisation de la classe de pseudonymisation et entrainement du modèle.
    ner = Ner(_inputs)
    corpus_with_labels = ner.predict_with_model(corpus, threshold)

    return corpus_with_labels


def evaluate_model(_inputs, corpus):
    """
    Evalue les performances du modèle indiqué dans _inputs sur un corpus donné.
    :param _inputs: paramètres du modèle à charger
    :param corpus: corpus sur lequel évaluer le modèle
    :return: statistiques d'évaluation du modèle
    """
    # Initialisation de la classe de pseudonymisation et entrainement du modèle.
    ner = Ner(_inputs)
    ner.evaluate_model(corpus)


def load_custom_corpus(list_of_texts, labels_format):
    """
    Convertit une liste de textes en un DataFrame au format attendu par l'algorithme de détection d'entités.
    :param list_of_texts: liste des textes à analyser
    :param labels_format: type de labels (BIO ou type d'entité uniquement)
    :return: DataFrame des textes
    """
    df = pd.DataFrame()
    texts = []
    texts_raw = []
    files = []

    for i, raw_text in enumerate(list_of_texts):
        annotations = pd.DataFrame(columns=["start_pos", "end_pos", "type", "word"])
        # Getting labels
        text, _ = get_text_and_labels(raw_text, annotations, labels_format)
        texts.append(text)
        texts_raw.append(raw_text)
        files.append(i)

    df['file'] = files
    df['raw_text'] = texts_raw
    df['text'] = texts

    return df


def load_doccano_corpus(list_of_corpus_path, labels_format):
    """
    Convertit un ou des corpus doccano au format jsonl en DataFrame(s) au format attendu par l'algorithme de détection
    d'entités.
    :param list_of_corpus_path: liste des chemins des corpus à charger
    :param labels_format: type de labels (BIO ou type d'entité uniquement)
    :return: liste des DataFrame correspondant aux corpus
    """
    corpus = {}
    for i, corpus_path in enumerate(list_of_corpus_path):

        _df = pd.DataFrame()
        texts = []
        texts_raw = []
        labels = []
        files = []
        df = pd.read_json(path_or_buf=corpus_path, lines=True)
        for _, row in df.iterrows():
            if len(row["label"]) > 0:
                annotations = pd.concat([pd.DataFrame([_ann + [row["text"][_ann[0]: _ann[1]]]],
                                                      columns=["start_pos", "end_pos", "type", "word"]) for _ann in
                                         row["label"]], ignore_index=True).sort_values(['start_pos'])
            else:
                annotations = pd.DataFrame(columns=["start_pos", "end_pos", "type", "word"])
            # Getting labels
            text, label = get_text_and_labels(row["text"], annotations, labels_format)
            texts.append(text)
            texts_raw.append(row["text"])
            labels.append(label)
            files.append(row["id"])

        _df['file'] = files
        _df['raw_text'] = texts_raw
        _df['text'] = texts
        _df['labels'] = labels
        corpus[i] = _df

    return list(corpus.values())


def replace_entities(corpus, entities, paths, labels_column, labels_format):
    """
    Remplace les entités précisées dans un corpus donné.
    :param corpus: DataFrame du corpus de textes
    :param entities: liste des entités à remplacer
    :param paths: chemins des fichiers csv utilisés pour générer des entités factices
    :param labels_column: nom de la colonne contenant les labels
    :param labels_format: type de labels (BIO ou type d'entité uniquement)
    :return: corpus avec entités remplacées
    """

    pseudo = Pseudo(paths["noms"], paths["adresses"], paths["vehicules"], paths["societes"], labels_column,
                    labels_format)
    corpus = pseudo.chain_of_replacements_other_entity(corpus, entities)

    return corpus


def show_annotations(corpus_df, entities, white_space_token, i=None):
    """
    Affiche des exemples de textes annotés dans un notebook.
    :param corpus_df: corpus de textes avec annotations
    :param entities: entités à afficher
    :param white_space_token: token du tokenizer correspondant à un espace (différents entre flauBERT et camemBERT)
    :param i: numéro du texte à afficher. Si non précisé, la fonction affiche 3 textes au hasard.
    """

    underlines = {ent: '#%02x%02x%02x' % (int(sns.color_palette('tab20', len(entities))[i][0] * 255),
                                          int(sns.color_palette('tab20', len(entities))[i][1] * 255),
                                          int(sns.color_palette('tab20', len(entities))[i][2] * 255))
                  for i, ent in enumerate(entities)}

    if i is not None:
        assert i <= len(corpus_df), "Le numéro de texte choisi ne doit pas excéder la taille du DataFrame."
        display(Markdown(f"# Texte n°{i}"))
        _sample = corpus_df.iloc[i:i + 1]
    else:
        display(Markdown(f"# Exemples de textes"))
        _sample = corpus_df[corpus_df[f"labels"].apply(lambda x: True if len(list(set(x))) > 1 else False)].sample(3)

    for _, _example in _sample.iterrows():
        text = ""
        _previous_label = "O"
        open_tag = 0
        for _word, _label in zip(_example["text"], _example[f"predicted_labels"]):
            _word = _word.replace(white_space_token, ' ')
            if _label.split('-')[-1] != _previous_label:
                if _label != "O":
                    if open_tag == 1:
                        text += '</code>'
                    open_tag = 1
                    entity = _label.split("-")[-1]
                    text += f'<code style="background:{underlines[entity]};color:black">'
                else:
                    open_tag = 0
                    text += '</code>'
            text += f'{_word}'
            _previous_label = _label.split('-')[-1]
        if open_tag == 1:
            text += '</code>'
        text += '<br>------------------------------------------<br>'

        display(Markdown(text))


def show_legend(entities):
    """
    Affiche dans un notebook la légende de couleurs des différentes entités (à appeler avec la fonction show_annotations
    ci-dessous).
    :param entities: liste des entités à afficher
    """
    underlines = {ent: '#%02x%02x%02x' % (int(sns.color_palette('tab20', len(entities))[i][0] * 255),
                                          int(sns.color_palette('tab20', len(entities))[i][1] * 255),
                                          int(sns.color_palette('tab20', len(entities))[i][2] * 255))
                  for i, ent in enumerate(entities)}
    legend = ""
    for ent in entities:
        legend += f'<code style="background:{underlines[ent]};color:black">{ent} </code>'
    display(Markdown(legend))


def train_and_evaluate_model(_inputs, train_corpus, test_corpus):
    """
    Entraine un modèle Transformers selon les paramètres précisés dans _inputs sur le corpus train_corpus et évalue ses
    performances sur le corpus test_corpus.
    :param _inputs: paramètres d'entrainement du modèle
    :param train_corpus: corpus d'entrainement
    :param test_corpus: corpus de test
    :return: statistiques d'évaluation du modèle
    """
    # Initialisation de la classe de pseudonymisation et entrainement du modèle.
    ner = Ner(_inputs)
    ner.get_corpus_stats(train_corpus)
    ner.train_model_on_corpus(train_corpus, test_corpus)


def write_doccano_format(corpus, white_space_token, output_file):
    """
    Exporte un DataFrame annoté au format doccano.
    :param corpus: DataFrame du corpus à exporter
    :param white_space_token: token du tokenizer correspondant à un espace (différents entre flauBERT et camemBERT)
    :param output_file: chemin du fichier de sortie
    """

    _n_start = 0
    _n_end = 0
    jsonl = []
    _full_json = {"id": [], "text": '', "label": []}
    _previous_part = 0
    for _, row in corpus.iterrows():
        labels = []
        _prev = "O"
        for i, _lab in enumerate(row["predicted_labels"]):
            if _lab != _prev:
                if _lab != "O":
                    if len(row["text"][i]) == 0:
                        _prefix = 0
                    else:
                        _prefix = 1 if row["text"][i][0] == white_space_token else 0
                    _n_start = len("".join(row["text"][:i]).replace(white_space_token, ' ')) + _prefix
                else:
                    _n_end = len("".join(row["text"][:i]).replace(white_space_token, ' '))
                    labels.append([_n_start, _n_end, _prev])
            _prev = _lab
        if _n_end < _n_start:
            labels.append([_n_start, len("".join(row["text"]).replace(white_space_token, ' ')), _prev])

        text_joined = "".join(row["text"]).replace(white_space_token, ' ')
        id_clean = row["file"].replace(white_space_token, ' ')
        _json = {"id": id_clean, "text": text_joined, "label": labels}
        # Merging parts of the same text together
        if row["text_part"] <= _previous_part:
            jsonl.append(_full_json)
            _full_json = {"id": _json["id"], "text": _json["text"], "label": _json["label"]}
        else:
            _full_json = {"id": _json["id"],
                          "text": _full_json["text"] + _json["text"],
                          "label": _full_json["label"] + _json["label"]}
        _previous_part = row["text_part"]
    jsonl.append(_full_json)

    with open(output_file, 'w', encoding='utf-8') as f:
        for file in jsonl:
            f.write(json.dumps(file) + '\n')
