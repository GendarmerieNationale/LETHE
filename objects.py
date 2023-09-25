import os, treetaggerwrapper, shutil, torch, rstr, datetime, re, unidecode, string, locale
import pandas as pd
import numpy as np
from random import random, randint, choice
from faker import Faker
from typing import List
from tqdm import tqdm
from datasets import Dataset
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from transformers import AutoModelForTokenClassification, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from utils import get_text_and_labels, split_into_parts, add_special_tokens_and_padding, extend_labels_to_tokens, \
    condense_labels, condense_labels_to_words, find_sub_list


class Corpus:
    """
    Class that allows loading and converting corpus annotated with different tools.
    """

    def __init__(self):

        self.corpus = {}
        self.annotation_labels = set()
        self.annotation_labels_map = {}

    def export_to_brat_annotations(self, corpus_key: str):
        """
        Export the chosen corpus to a list of dictionnaries with BRAT formatting that can be easily saved as a list of
        .txt ad .ann files.
        """

        files_name, ann_files, txt_files = ([] for _ in range(3))
        _corpus = self.get_corpus_from_key(corpus_key)

        for _, row in _corpus.iterrows():

            files_name.append(row["file"])
            txt_files.append(" ".join(row["words"]))
            labels = [{"value": {
                "start": len(" ".join(row["words"][:j])) + 1,
                "end": len(" ".join(row["words"][:j])) + len(row["words"][j]) + 1,
                "text": " ".join(row["words"])[
                        len(" ".join(row["words"][:j])) + 1:len(" ".join(row["words"][:j])) + len(row["words"][j]) + 1],
                "labels": [row["words_labels"][j]]
            }
            } for j in range(len(row.words_labels)) if row["words_labels"][j] != "O"]
            labels = condense_labels(labels)
            file = []
            for i, label in enumerate(labels):
                _label = label["value"]["labels"][0]
                _word = label["value"]["text"]
                _start = label["value"]["start"]
                _end = label["value"]["end"]
                file += [f"T{i+1}\t{_label} {_start} {_end}\t{_word}"]
            ann_files.append(file)
        return [{"id": name, "txt": txt, "ann": ann} for name, txt, ann in zip(files_name, txt_files, ann_files)]

    def export_to_doccano_annotations(self, corpus_key: str):
        """
        Export the chosen corpus to a list of dictionnaries with Doccano formatting that can be easily saved to a .jsonl
        file.
        """

        dc_json = []
        _corpus = self.get_corpus_from_key(corpus_key)
        for i, row in _corpus.iterrows():
            labels = [{"value": {
                    "start": len(" ".join(row["words"][:j])) + 1,
                    "end": len(" ".join(row["words"][:j])) + len(row["words"][j]) + 1,
                    "text": " ".join(row["words"])[
                            len(" ".join(row["words"][:j])) + 1:len(" ".join(row["words"][:j])) + len(
                                row["words"][j]) + 1],
                    "labels": [row["words_labels"][j]]
                }
            } for j in range(len(row.words_labels)) if row["words_labels"][j] != "O"]
            labels = condense_labels(labels)
            labels = [[x["value"]["start"], x["value"]["end"], x["value"]["labels"][0]] for x in labels]
            line = {"id": i, "text": " ".join(row["words"]), "label": labels}
            dc_json.append(line)
        return dc_json

    def export_to_labelstudio_annotations(self, corpus_key: str):
        """
        Export the chosen corpus to a list of dictionnaries with labelStudio formatting that can be easily saved to a
        .jsonl file.
        """

        ls_json = []
        date_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        _corpus = self.get_corpus_from_key(corpus_key)
        for i, row in _corpus.iterrows():
            results = [{"value":
                        {
                            "start": len(" ".join(row["words"][:j])) + 1,
                            "end": len(" ".join(row["words"][:j])) + len(row["words"][j]) + 1,
                            "text": row["words"][j],
                            "labels": [row["words_labels"][j]]
                        },
                        # Random id of 10 characters
                        "id": rstr.xeger(r'[a-zA-Z]{10}'),
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "origin": "manual"
                        } for j in range(len(row.words_labels)) if row["words_labels"][j] != "O"]
            results = condense_labels(results)
            _entry = {
                "id": i,
                "annotations": [
                    {
                        "id": 1,
                        "completed_by": 1,
                        "result": results,
                        "was_cancelled": False,
                        "ground_truth": False,
                        "created_at": date_str,
                        "updated_at": date_str,
                        "lead_time": 3.095,
                        "prediction": {},
                        "result_count": 0,
                        # Random id with specific format
                        "unique_id": rstr.xeger(r'[a-z0-9]{8}-([a-z0-9]{4}-){3}[a-zA-Z0-9]{12}'),
                        "last_action": None,
                        "task": 7,
                        "project": 2,
                        "updated_by": 1,
                        "parent_prediction": None,
                        "parent_annotation": None,
                        "last_created_by": None
                    }
                ],
                "drafts": [],
                "predictions": [],
                "data": {"text": " ".join(row["words"])},
                "meta": {},
                "created_at": date_str,
                "updated_at": date_str,
                "inner_id": i,
                "total_annotations": len(results),
                "cancelled_annotations": 0,
                "total_predictions": 0,
                "comment_count": 0,
                "unresolved_comment_count": 0,
                "last_comment_updated_at": None,
                "project": 2,
                "updated_by": 1,
                "comment_authors": []
            }
            ls_json.append(_entry)
        return ls_json

    def export_to_flair_dataset(self, corpus_dict: dict):
        """
        Export the chosen corpus to a flair Dataset ColumnCorpus object that can be used to train a model.
        """

        if not os.path.exists("flair_temp"):
            os.mkdir("flair_temp")

        columns = {0: "text", 1: "pos", 2: "ner"}
        for function, key in corpus_dict.items():
            if key is None:
                open(f"flair_temp/{function}.txt", "w")
                continue
            _flair_corpus = []
            _corpus = self.get_corpus_from_key(key)
            if len(_corpus) > 0:
                for _, row in _corpus.iterrows():
                    tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr')
                    pos_tags = [x.split("\t")[1] if len(x.split("\t")) == 3 else "?" for x in
                                tagger.tag_text(row["words"], tagonly=True)]
                    pos_tags = [x.split(":")[0] for x in pos_tags]
                    annotated_text = [" ".join(x) for x in zip(row["words"], pos_tags, row["words_labels"])]
                    _flair_corpus += annotated_text + ["\n"]

            f = open(f"flair_temp/{function}.txt", "w")
            for line in _flair_corpus:
                f.write(line)
                f.write("\n")
            f.close()

        corpus = ColumnCorpus("flair_temp/", columns, train_file="train.txt", test_file="test.txt", dev_file="dev.txt")
        shutil.rmtree("flair_temp/")

        return corpus

    def export_to_hugginface_dataset(self, max_seq_length: int, tokenizer_config: dict, special_tokens: dict,
                                     corpus_type: str, model_columns: List[str]):
        """
        Export the chosen corpus to a hugginface Dataset object that can be used to train a model.
        """

        _corpus = self.get_corpus_from_key(corpus_type)
        self.annotation_labels_map.update({"": special_tokens["pad_label_id"]})
        special_tokens_count = 3 if tokenizer_config["extra_sep"] else 2

        features = pd.DataFrame()
        for i_index, row in _corpus.iterrows():
            labels_ids = [self.annotation_labels_map[label] for label in row["tokens_labels"]]
            # Stores texts and labels into InputFeatures objects
            chunks = split_into_parts(row["tokens_ids"], labels_ids, row["tokens_ids_file"],
                                      max_seq_length - special_tokens_count)
            # Adding special tokens and padding and creating mask and segment
            for chunk in chunks:
                text_tokens_ids, mask_ids, label_ids, file_tokens_ids = \
                    add_special_tokens_and_padding(chunk["tokens_ids"], chunk["labels_ids"], chunk["tokens_ids_file"],
                                                   tokenizer_config, special_tokens, max_seq_length)
                _df = pd.DataFrame({"input_ids": [text_tokens_ids], "attention_mask": [mask_ids],
                                    "labels": [label_ids]})
                _df.columns = model_columns
                features = pd.concat([features, _df])

        dataset = Dataset.from_pandas(features)

        return dataset

    def export_to_torch_tensordataset(self, max_seq_length: int, tokenizer_config: dict, special_tokens: dict,
                                      corpus_type: str):
        """
        Export the chosen corpus to a pytorch TensorDataset object that can be used to train a model.
        """

        _corpus = self.get_corpus_from_key(corpus_type)
        self.annotation_labels_map.update({"": special_tokens["pad_label_id"]})
        special_tokens_count = 3 if tokenizer_config["extra_sep"] else 2

        features = []
        for i_index, row in _corpus.iterrows():
            labels_ids = [self.annotation_labels_map[label] for label in row["tokens_labels"]]
            # Stores texts and labels into InputFeatures objects
            chunks = split_into_parts(row["tokens_ids"], labels_ids, row["tokens_ids_file"],
                                      max_seq_length - special_tokens_count)
            # Adding special tokens and padding and creating mask and segment
            for chunk in chunks:
                text_tokens_ids, mask_ids, label_ids, file_tokens_ids = \
                    add_special_tokens_and_padding(chunk["tokens_ids"], chunk["labels_ids"], chunk["tokens_ids_file"],
                                                   tokenizer_config, special_tokens, max_seq_length)
                features.append({"text_token_ids": text_tokens_ids, "text_mask": mask_ids,
                                 "label_ids": label_ids, "text_part_index": chunk["part_index"],
                                 "file_token_ids": file_tokens_ids})

        # Convert to Tensors and build dataset
        all_text_token_ids = torch.tensor([f["text_token_ids"] for f in features], dtype=torch.long)
        all_text_mask = torch.tensor([f["text_mask"] for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f["label_ids"] for f in features], dtype=torch.long)
        all_text_parts_ids = torch.tensor([f["text_part_index"] for f in features], dtype=torch.long)
        all_file_token_ids = torch.tensor([f["file_token_ids"] for f in features], dtype=torch.long)

        dataset = TensorDataset(all_text_token_ids, all_text_mask, all_label_ids, all_text_parts_ids,
                                all_file_token_ids)
        return dataset

    def filter_labels(self, labels: List[str]):
        """
        Filter all the labels absent from the specified list by replacing them by "O".
        :param labels: list of labels to keep
        """

        for _corpus in self.corpus.values():
            if "words_labels" in _corpus.columns:
                _corpus["words_labels"] = _corpus["words_labels"].apply(
                    lambda x: [lab if lab in labels else "O" for lab in x])
        self.annotation_labels = set(labels + ["O"])
        self.annotation_labels_map = {label: i for i, label in enumerate(self.annotation_labels)}

    def get_corpus_from_key(self, corpus_key: str):
        """
        Returns a corpus from its key.
        :param corpus_key: corpus key
        return: corpus (DataFrame)
        """

        assert corpus_key in self.corpus.keys(), "Invalid corpus_key"

        return self.corpus[corpus_key]

    def get_labels(self):
        """
        Gets the list of all the different annotations that appear in the corpus.
        return: list of annotations.
        """

        if all("words_labels" not in _corpus.columns for _corpus in self.corpus.values()):
            return []
        else:
            labels = set()
            for _corpus in self.corpus.values():
                if "words_labels" in _corpus.columns:
                    labels.update(set([y for x in _corpus["words_labels"] for y in x]))
            return list(labels)

    def load_brat_annotations(self, ann_files_directory: str, labels_format: str, corpus_key: str = "train"):
        """
        Loads annotations created with BRAT (from a folder containing .txt and .ann files) and links them to the
        specified corpus key.
        :param ann_files_directory: folder containing BRAT annotation files
        :param labels_format: format of the labels
        :param corpus_key: corpus key
        """

        self.corpus[corpus_key] = pd.DataFrame()
        _corpus = self.corpus[corpus_key]

        texts, words, labels, files = ([] for _ in range(4))
        _ann_files = [file for file in os.listdir(ann_files_directory) if file.endswith(".ann")]
        for file in _ann_files:
            ann_path = os.path.join(ann_files_directory, file)
            txt_path = os.path.join(ann_files_directory, file).replace(".ann", ".txt")
            with open(txt_path, errors='ignore') as f:
                texte = f.read()
            texte = texte.replace("\n", " ")

            try:
                ann = pd.read_csv(ann_path, sep="\t", header=None)
                ann["type"] = ann[1].apply(lambda x: x.split(" ")[0])
                ann["start_pos"] = ann[1].apply(lambda x: int(x.split(" ")[1]))
                ann["end_pos"] = ann[1].apply(lambda x: int(x.split(" ")[2]))
                ann = ann[["start_pos", "end_pos", "type", 2]].sort_values(["start_pos"]).rename(columns={2: "word"})
            except:
                ann = pd.DataFrame(columns=["word", "type", "start_pos", "end_pos"])

            # Getting labels
            clean_text = texte.replace("\n", " ")
            word, label = get_text_and_labels(clean_text, ann, labels_format)
            words.append(word)
            texts.append(texte)
            labels.append(label)
            files.append(file.replace(".ann", ""))

        _corpus['file'] = files
        _corpus['text'] = texts
        _corpus['words'] = words
        _corpus['words_labels'] = labels

        new_labels = set([y for x in _corpus["words_labels"] for y in x])
        self.annotation_labels.update(new_labels)
        self.annotation_labels_map = {label: i for i, label in enumerate(self.annotation_labels)}

    def load_doccano_annotations(self, jsonl_annotation_path: str, labels_format: str, corpus_key: str = "train"):
        """
        Loads annotations created with Doccano (a jsonl file) and links them to the specified corpus key.
        :param jsonl_annotation_path: Doccano annotation files
        :param labels_format: format of the labels
        :param corpus_key: corpus key
        """
        self.corpus[corpus_key] = pd.DataFrame()
        _corpus = self.corpus[corpus_key]

        texts, words, labels, files = ([] for _ in range(4))
        annotations = pd.read_json(path_or_buf=jsonl_annotation_path, lines=True)
        for _, row in annotations.iterrows():
            if len(row["label"]) > 0:
                annotations = pd.concat([pd.DataFrame([_ann + [row["text"][_ann[0]: _ann[1]]]],
                                                      columns=["start_pos", "end_pos", "type", "word"]) for _ann in
                                         row["label"]], ignore_index=True).sort_values(['start_pos'])
            else:
                annotations = pd.DataFrame(columns=["start_pos", "end_pos", "type", "word"])
            # Getting labels
            clean_text = row["text"].replace("\n", " ")
            word, label = get_text_and_labels(clean_text, annotations, labels_format)
            words.append(word)
            texts.append(clean_text)
            labels.append(label)
            files.append(row["id"])

        _corpus['file'] = files
        _corpus['text'] = texts
        _corpus['words'] = words
        _corpus['words_labels'] = labels

        new_labels = set([y for x in _corpus["words_labels"] for y in x])
        self.annotation_labels.update(new_labels)
        self.annotation_labels_map = {label: i for i, label in enumerate(self.annotation_labels)}

    def load_labelstudio_annotations(self, jsonl_annotation_path: str, labels_format: str, corpus_key: str = "train"):
        """
        Loads annotations created with LabelStudio (a jsonl file) and links them to the specified corpus key.
        :param jsonl_annotation_path: Doccano annotation files
        :param labels_format: format of the labels
        :param corpus_key: corpus key
        """
        self.corpus[corpus_key] = pd.DataFrame()
        _corpus = self.corpus[corpus_key]

        texts, words, labels, files = ([] for _ in range(4))
        annotations = pd.read_json(path_or_buf=jsonl_annotation_path, lines=False)
        for _, row in annotations.iterrows():
            start_pos = [y["value"]["start"] for x in row["annotations"] for y in x["result"]]
            end_pos = [y["value"]["end"] for x in row["annotations"] for y in x["result"]]
            _type = [y["value"]["labels"][0] for x in row["annotations"] for y in x["result"]]
            word = [y["value"]["text"] for x in row["annotations"] for y in x["result"]]
            annotations = pd.DataFrame({"start_pos": start_pos, "end_pos": end_pos, "type": _type, "word": word})
            # Getting labels
            clean_text = row["data"]["text"].replace("\n", " ")
            word, label = get_text_and_labels(clean_text, annotations, labels_format)
            words.append(word)
            texts.append(clean_text)
            labels.append(label)
            files.append(row["id"])

        _corpus['file'] = files
        _corpus['text'] = texts
        _corpus['words'] = words
        _corpus['words_labels'] = labels

        new_labels = set([y for x in _corpus["words_labels"] for y in x])
        self.annotation_labels.update(new_labels)
        self.annotation_labels_map = {label: i for i, label in enumerate(self.annotation_labels)}

    def load_list_of_texts(self, texts: List[str], corpus_key: str = "train", labels=None):
        """
        Loads a specific list of texts and links them to the specified corpus key.
        :param texts: list of texts as strings
        :param corpus_key: corpus key
        :param labels: labels corresponding to the texts (optional)
        """
        self.corpus[corpus_key] = pd.DataFrame()
        _corpus = self.corpus[corpus_key]

        _corpus['file'] = [i for i in range(len(texts))]
        _corpus['text'] = [x.replace("\n", " ") for x in texts]
        _corpus['words'] = [x.split(" ") for x in texts]
        if labels is None:
            _corpus['words_labels'] = _corpus['words'].apply(lambda x: ["O" for _ in x])
        else:
            _corpus['words_labels'] = labels

        new_labels = set([y for x in _corpus["words_labels"] for y in x])
        self.annotation_labels.update(new_labels)
        self.annotation_labels_map = {label: i for i, label in enumerate(self.annotation_labels)}

    def tokenize_and_keep_labels(self, tokenizer, corpus_key: str, method="full"):
        """
        Converts the texts of a specific corpus to tokens.
        :param tokenizer: tokenizer to use
        :param corpus_key: corpus key
        :param method: defines how to label tokens issued from a single word:
            - "full": all the tokens take the word label.
            - "first": only the first token takes the word label.
        """

        _corpus = self.get_corpus_from_key(corpus_key)

        _corpus["tokens_file"] = _corpus["file"].apply(lambda _id: tokenizer.tokenize(str(_id)))
        _corpus["tokens"] = _corpus["words"].apply(
            lambda words: [tokenizer.tokenize(word) for word in words])
        _corpus["tokens_labels"] = _corpus[["tokens", "words_labels"]].apply(lambda r: [y for x in [
            extend_labels_to_tokens(lab, len(tokens), method) for tokens, lab in zip(r["tokens"], r["words_labels"])]
                                                                                        for y in x], axis=1)
        _corpus["tokens"] = _corpus["tokens"].apply(lambda tokens: [y for x in tokens for y in x])
        _corpus["tokens_ids_file"] = _corpus["tokens_file"].apply(
            lambda x: [tokenizer.convert_tokens_to_ids(token) for token in x])
        _corpus["tokens_ids"] = _corpus["tokens"].apply(
            lambda x: [tokenizer.convert_tokens_to_ids(token) for token in x])


class EntityDetector:
    """
    Class to detect entities in a given Corpus using different kinds of models.
    """

    def __init__(self, model_type: str, model_path: str, model_labels: List[str], pad_label_id: [int, None] = None,
                 tokenizer_config: [dict, None] = None):

        self.model_type = model_type
        self.model_labels = model_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model_type == "flair":
            self.tagger = SequenceTagger.load(model_path)
        elif model_type == "hugginface":
            self.tagger = AutoModelForTokenClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer_config = tokenizer_config
            self.special_tokens = {"cls_id": self.tokenizer.cls_token_id, "sep_id": self.tokenizer.sep_token_id,
                                   "pad_id": self.tokenizer.pad_token_id, "pad_label_id": pad_label_id}
        self.tagger.to(self.device)
        self.tagger.eval()
        self.regex_functions = {"TIME": self.regex_time, "PHONE": self.regex_phone, "EMAIL": self.regex_email,
                                "IMMAT": self.regex_immat}

    @staticmethod
    def combine_labels(old_labels, new_labels):
        """
        Adds new predictions to labels.
        :param old_labels: labels before the predictions.
        :param new_labels: new detected entities.
        return: labels combining previous and new predictions.
        """

        assert len(old_labels) == len(new_labels), "The number of labels must be equal."
        combined_labels = [_new if _new != "O" else _old for _old, _new in zip(old_labels, new_labels)]
        return combined_labels

    def process_corpus_flair(self, corpus: Corpus, corpus_key: str, batch_size: int = 12):
        """
        Entity detection using a Flair model.
        :param corpus: Corpus object
        :param corpus_key: key of the corpus where we want to find entities
        :param batch_size: batch size for inference
        """

        dataset = corpus.export_to_flair_dataset({"train": corpus_key, "test": None, "dev": None})
        _corpus = getattr(dataset, "train")

        self.tagger.predict(_corpus, batch_size, return_probabilities_for_all_classes=True)
        entities = [x.get_spans("ner") for x in _corpus.datasets[0].sentences]
        _flair_words_labels = []
        all_probas = []
        for i, row in corpus.corpus[corpus_key].iterrows():
            predictions = ["O" for _ in row["words"]]
            _pos = {"start": [len(" ".join(row["words"][:i-1])) + 1 for i in range(1, len(row["words"]))],
                    "end": [len(" ".join(row["words"][:i])) for i in range(1, len(row["words"]))]}
            _pos["start"][0] = 0
            for ent in entities[i]:
                _id_start = _pos["start"].index(ent.start_position)
                _id_end = _id_start + _pos["end"][_id_start:].index(ent.end_position)
                if " ".join(row["words"][_id_start: _id_end + 1]) == ent.text:
                    predictions[_id_start: _id_end + 1] = [ent.tag for _ in range(_id_start, _id_end + 1)]
            probas = [max(x.tags_proba_dist["ner"], key=lambda y: y.score).score for x in
                      _corpus.datasets[0].sentences[i].tokens]
            _flair_words_labels.append(predictions)
            all_probas.append(probas)

        if "predicted_words_labels" not in corpus.corpus[corpus_key].columns:
            corpus.corpus[corpus_key]["predicted_words_labels"] = corpus.corpus[corpus_key]["words"].apply(
                lambda x: ["O" for _ in x])
        corpus.corpus[corpus_key]["predicted_words_labels"] = [self.combine_labels(x, y) for x, y in
                                                               zip(corpus.corpus[corpus_key]["predicted_words_labels"],
                                                                   _flair_words_labels)]
        corpus.corpus[corpus_key]["predictions_probabilities"] = all_probas

    def process_corpus_hugginface(self, corpus: Corpus, corpus_key: str, batch_size: int = 12,
                                  max_seq_length: int = 128, threshold: [dict, None] = None):
        """
        Entity detection using a hugginface model.
        :param corpus: Corpus object
        :param corpus_key: key of the corpus where we want to find entities
        :param batch_size: batch size for inference
        :param max_seq_length: maximum number of tokens
        :param threshold: thresholds associated with each entity. If the probability of a given label exceeds its
        threshold, it will be the one selected even if there are entities with higher probabilities.
        """

        model_label_map = {i: label for i, label in enumerate(self.model_labels)}
        annotation_label_map = {v: k for k, v in corpus.annotation_labels_map.items()}
        if threshold is not None:
            threshold = {ind: threshold[ent] if ent in threshold.keys() else 1000 for ind, ent in
                         model_label_map.items()}
        corpus.tokenize_and_keep_labels(self.tokenizer, corpus_key, method="full")
        dataset = corpus.export_to_torch_tensordataset(max_seq_length, self.tokenizer_config, self.special_tokens,
                                                       corpus_key)
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

        all_predictions = pd.DataFrame()
        for batch in tqdm(dataloader, desc="Evaluating", position=0, leave=True):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                _inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}
                outputs = self.tagger(**_inputs)
                _output_probabilities = outputs[:2][0]

            files, text_parts, text_tokens, real_labels, predicted_labels, labels_probabilities = \
                self.extract_info_from_batch(self.tokenizer, batch, _output_probabilities, model_label_map,
                                             annotation_label_map, threshold)

            predictions = pd.DataFrame({"file": files, "text_part": text_parts, "text": text_tokens,
                                        "labels": real_labels, "predicted_labels": predicted_labels,
                                        "labels_probabilities": labels_probabilities})
            predictions = predictions.sort_values(["file", "text_part"]).groupby("file")\
                .agg({'predicted_labels': "sum", 'labels_probabilities': "sum"})
            all_predictions = pd.concat([all_predictions, predictions])

        if "predicted_words_labels" not in corpus.corpus[corpus_key].columns:
            corpus.corpus[corpus_key]["predicted_words_labels"] = corpus.corpus[corpus_key]["words"].apply(
                lambda x: ["O" for _ in x])
        corpus.corpus[corpus_key]["predicted_tokens_labels"] = all_predictions["predicted_labels"].tolist()
        corpus.corpus[corpus_key]["predictions_probabilities"] = all_predictions["labels_probabilities"].tolist()
        _hugginface_words_labels = corpus.corpus[corpus_key][
            ["tokens", "predicted_tokens_labels"]].apply(
            lambda row: condense_labels_to_words(row["tokens"], row["predicted_tokens_labels"]), axis=1)
        corpus.corpus[corpus_key]["predicted_words_labels"] = self.combine_labels(
            corpus.corpus[corpus_key]["predicted_words_labels"], _hugginface_words_labels)

    def process_corpus_regex(self, corpus: Corpus, corpus_key: str):
        """
        Uses regular expressions to detect entities in a specific Corpus. The functions used are the ones defined
        in the variable regex_functions initialized when calling EntityDetector.
        :param corpus: Corpus object
        :param corpus_key: key of the corpus where we want to find entities
        """

        _corpus = corpus.corpus[corpus_key]
        if "predicted_words_labels" not in _corpus.columns:
            _corpus["predicted_words_labels"] = _corpus["words"].apply(lambda x: ["O" for _ in x])

        for regex_function in self.regex_functions.values():
            _corpus["predicted_words_labels"] = _corpus[["text", "words", "predicted_words_labels"]].apply(
                lambda x: regex_function(x), axis=1)

    def extract_info_from_batch(self, tokenizer, batch: tuple, _output_probabilities: torch.Tensor,
                                model_label_map: dict, annotation_label_map: dict, threshold: [dict, None] = None):
        """
        Extracts relevant information from a batch object.
        :param tokenizer: tokenizer
        :param batch: batch
        :param _output_probabilities: probabilities of the different labels
        :param model_label_map: mapping between labels and their id.
        :param annotation_label_map: mapping between annotation labels and their id.
        :param threshold: thresholds associated with each entity. If the probability of a given label exceeds its
        threshold, it will be the one selected even if there are entities with higher probabilities.
        :return: files names, text_parts, text tokens, real labels, predicted labels and  labels probabilities for all
        the texts of the batch.
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

        predicted_labels = [[model_label_map[x] for x in y] for y in predicted_labels_ids]
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
        real_labels = [[annotation_label_map[int(x)] for x in y if x != self.special_tokens["pad_label_id"]]
                       for y in batch[2]]
        real_labels = [x for i, x in enumerate(real_labels) if i in _valid_examples]
        # Extract file names
        file_tokens = [[tokenizer.convert_ids_to_tokens(int(x)) for x in y if x not in token_ids_2_ignore] for y in
                       batch[4]]
        files = ["".join([x.replace('▁', ' ') for x in y]).strip() for y in file_tokens]
        files = [x for i, x in enumerate(files) if i in _valid_examples]
        # Extract text part
        text_parts = [int(x) for x in batch[3]]
        text_parts = [x for i, x in enumerate(text_parts) if i in _valid_examples]

        return files, text_parts, text_tokens, real_labels, predicted_labels, labels_probabilities

    def regex_immat(self, row):
        """
        Finds immats in texts using REGEX rules.
        :param row: DataFrame row
        :return: pd.Series with file, text and label columns
        """

        tag_name = [k for k, v in self.regex_functions.items() if v == self.regex_email][0]
        raw_ppel = row["text"]

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

        plaque = sorted(plaque, key=lambda x: x[0])

        _words_pos = [(len(" ".join(row["words"][:i - 1])), len(" ".join(row["words"][:i]))) for i in
                      range(1, len(row["words"]) + 1)]

        _regex_labels = ["O" for _ in row["words"]]
        for i, (_s, _e) in enumerate(_words_pos):
            _characters = set(range(_s, _e))
            if any([len(_characters.intersection(set(range(_s, _e)))) > 0 for _s, _e, _ in plaque]):
                _regex_labels[i] = tag_name

        return self.combine_labels(row["predicted_words_labels"], _regex_labels)

    def regex_email(self, row):
        """
        Finds e-mails in texts using REGEX rules.
        :param row: DataFrame row
        :return: pd.Series with file, text and label columns
        """

        tag_name = [k for k, v in self.regex_functions.items() if v == self.regex_email][0]
        raw_ppel = row["text"]

        # REGEX time patterns
        regex_pattern = r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"

        # Finds e-mail patterns
        emails = []
        for _mail in re.finditer(regex_pattern, raw_ppel):
            s = _mail.start()
            e = _mail.end()
            if raw_ppel[e - 1] == '.':
                emails.append((s, e, raw_ppel[s:e - 1]))
            else:
                emails.append((s, e, raw_ppel[s:e]))

        emails = sorted(emails, key=lambda x: x[0])

        _words_pos = [(len(" ".join(row["words"][:i - 1])), len(" ".join(row["words"][:i]))) for i in
                      range(1, len(row["words"]) + 1)]

        _regex_labels = ["O" for _ in row["words"]]
        for i, (_s, _e) in enumerate(_words_pos):
            _characters = set(range(_s, _e))
            if any([len(_characters.intersection(set(range(_s, _e)))) > 0 for _s, _e, _ in emails]):
                _regex_labels[i] = tag_name

        return self.combine_labels(row["predicted_words_labels"], _regex_labels)

    def regex_phone(self, row):
        """
        Finds phone numbers in texts using REGEX rules.
        :param row: DataFrame row
        :return: pd.Series with file, text and label columns
        """

        tag_name = [k for k, v in self.regex_functions.items() if v == self.regex_phone][0]
        raw_ppel = row["text"]

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
                phones.append((s, e, raw_ppel[s:e].strip()))
        phones = sorted(phones, key=lambda x: x[0])

        _words_pos = [(len(" ".join(row["words"][:i - 1])), len(" ".join(row["words"][:i]))) for i in
                      range(1, len(row["words"]) + 1)]

        _regex_labels = ["O" for _ in row["words"]]
        for i, (_s, _e) in enumerate(_words_pos):
            _characters = set(range(_s, _e))
            if any([len(_characters.intersection(set(range(_s, _e)))) > 0 for _s, _e, _ in phones]):
                _regex_labels[i] = tag_name

        return self.combine_labels(row["predicted_words_labels"], _regex_labels)

    def regex_time(self, row):
        """
        Finds times in texts using REGEX rules.
        :param row: DataFrame row
        :return: pd.Series with file, text and label columns
        """

        tag_name = [k for k, v in self.regex_functions.items() if v == self.regex_time][0]
        raw_ppel = row["text"]

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
                times.append((s, e, raw_ppel[s:e].strip()))
        times = sorted(times, key=lambda x: x[0])

        _words_pos = [(len(" ".join(row["words"][:i - 1])), len(" ".join(row["words"][:i]))) for i in
                      range(1, len(row["words"]) + 1)]

        _regex_labels = ["O" for _ in row["words"]]
        for i, (_s, _e) in enumerate(_words_pos):
            _characters = set(range(_s, _e))
            if any([len(_characters.intersection(set(range(_s, _e)))) > 0 for _s, _e, _ in times]):
                _regex_labels[i] = tag_name

        return self.combine_labels(row["predicted_words_labels"], _regex_labels)


class Pseudo:
    """
    Class to replace detected entities by fake ones of tags.
    """

    def __init__(self, _names_path: str, _address_path: str, _car_path: str, societies_path: str, labels_format: str,
                 white_space_token: str):

        self.entity_functions = {"REF_NUM": self.replace_refnum, "LOC": self.replace_loc, "PERSON": self.replace_person,
                                 "ORGANIZATION": self.replace_organization, "WEBSITE": self.replace_website,
                                 "ADDRESS": self.replace_address, "ZIP": self.replace_zip,
                                 "CITY": self.replace_cities, "EMAIL": self.replace_email,
                                 "PHONE": self.replace_phone, "IMMAT": self.replace_immat, "MONEY": self.replace_money,
                                 "DATE": self.replace_date, "TIME": self.replace_time, "CAR": self.replace_car}
        self.names_path = _names_path
        self.address_path = _address_path
        self.car_path = _car_path
        self.societies_path = societies_path
        self.labels_format = labels_format
        self.fake = Faker('fr_FR')
        self.white_space_token = white_space_token
        Faker.seed()

        self.address, self.names, self.zip, self.cars, self.societies, self.train_df, self.dev_df, self.test_df = \
            [None] * 8

    def replace_entities_by_fake_ones(self, corpus: Corpus, corpus_key: str, list_entities: List[str],
                                      labels_column: str):
        """
        Replaces all the entities of a given list by fake ones of the same type in a specific corpus.
        :param corpus: Corpus object
        :param corpus_key: key of the corpus where we want to replace entities
        :param list_entities: list of entity types to replace
        :param labels_column: labels column name
        """

        self.address = pd.read_csv(self.address_path)
        self.names = pd.read_csv(self.names_path)
        self.zip = self.address['postcode'].unique().tolist()
        self.cars = pd.read_csv(self.car_path)
        self.societies = pd.read_csv(self.societies_path)

        _corpus = corpus.corpus[corpus_key]
        _corpus["words_pseudonymized"] = _corpus["words"].copy()
        _corpus["labels_pseudonymized"] = _corpus[labels_column].copy()
        for ent in list_entities:
            if ent not in self.entity_functions.keys():
                print(f"Entity {ent} is unknown. Write your own replacement function for this specific entity.")
                continue
            print(f"Replacing entity: {ent}")
            results = _corpus[["words_pseudonymized", "labels_pseudonymized"]]\
                .apply(lambda x: self.entity_functions[ent](x), axis=1)
            _corpus["words_pseudonymized"] = [x[0] for x in results]
            _corpus["labels_pseudonymized"] = [x[1] for x in results]
        _corpus["text_pseudonymized"] = _corpus["words_pseudonymized"].apply(lambda x: " ".join(x))

    def replace_entities_by_tags(self, corpus: Corpus, corpus_key: str, list_entities: List[str], labels_column: str):
        """
        Replaces all the entities of a given list by tags in a specific corpus.
        :param corpus: Corpus object
        :param corpus_key: key of the corpus where we want to replace entities
        :param list_entities: list of entity types to replace
        :param labels_column: labels column name
        """

        _corpus = corpus.corpus[corpus_key]
        _corpus["words_pseudonymized"] = _corpus["words"].copy()
        _corpus["labels_pseudonymized"] = _corpus[labels_column].copy()
        for entity in list_entities:
            results = _corpus[["words_pseudonymized", "labels_pseudonymized"]]\
                .apply(lambda x: self.replace_by_tag(x, entity), axis=1)
            _corpus["words_pseudonymized"] = [x[0] for x in results]
            _corpus["labels_pseudonymized"] = [x[1] for x in results]
        _corpus["text_pseudonymized"] = _corpus["words_pseudonymized"].apply(lambda x: " ".join(x))

    def concat_entities(self, found_list: List[tuple]):
        """
        Concatenates entities splitted into several successive tokens in a single entity.
        :param found_list: list of entities, each element being (index, word, label)
        :return: concatenated entities
        """

        clean_list = []
        if found_list:
            full_entity = found_list[0][1]
            for i in range(1, len(found_list)):
                if self.labels_format == "BIO":
                    _is_same_entity = (found_list[i][0] == found_list[i - 1][0] + 1) & (found_list[i][2] == 'I')
                else:
                    _is_same_entity = (found_list[i][0] == found_list[i - 1][0] + 1)
                if _is_same_entity and (not found_list[i-1][1].endswith('.')):
                    full_entity += ' ' + found_list[i][1]
                else:
                    clean_list.append(full_entity)
                    full_entity = found_list[i][1]
            clean_list.append(full_entity)
        else:
            clean_list = []

        return clean_list

    def create_csv_name(self, _name: str):
        """
        Creates a fake name from a csv database.
        :param _name: name to replace
        :return: fake name
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

    def create_faker_name(self, _name: str):
        """
        Creates a fake name using Faker library.
        :param _name: name to replace
        :return: fake name
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
    def match_case(new_entity: str, old_entity: str):
        """
        Matches a fake generated entity with the "style" of the initial one (capital letters or not).
        :param new_entity: fake generated entity
        :param old_entity: initial entity
        :return: fake entity with the right "style"
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
    def match_punctuation(new_entity: str, old_entity: str):
        """
        Adds punctuation at the end of the fake generated entity if there was one initially.
        :param new_entity: fake generated entity
        :param old_entity: initial entity
        :return: fake entity with the right ending punctuation
        """

        # looking for possible endings to reproduce
        endings = ['.<br>', ',<br>', ',', '.']
        for _ending in endings:
            if (old_entity[-len(_ending):] == _ending) and (new_entity[-len(_ending):] != _ending):
                new_entity += _ending
        return new_entity

    def replace_by_tag(self, row: pd.Series, entity: str):
        """
        Replaces an entity by a tag. The tag format is "<entity_name>_<entity_type_number_in_the_text>".
        :param row: DataFrame row
        :param entity: entity to replace
        :return: words and labels with replaced entity. We need to update the labels as well because the initial and
        fake entities can have different number of words.
        """

        entities = [(ind, row["words_pseudonymized"][ind], entity.split('-')[0]) for ind, ent in
                    enumerate(row["labels_pseudonymized"]) if ent.split('-')[-1] == entity]
        clean_entities = self.concat_entities(entities)
        replacement_tags = {y: f"<{entity}_{i + 1}>" for i, y in enumerate(list(dict.fromkeys(clean_entities)))}

        replaced_words = row["words_pseudonymized"].copy()
        labels = row["labels_pseudonymized"].copy()
        for _ent in clean_entities:
            replaced_words, labels = self.switch_entity(replaced_words, labels, _ent, replacement_tags[_ent], entity)

        return replaced_words, labels

    def replace_person(self, row: pd.Series, method: str = "dataset"):
        """
        Replaces names by fake ones.
        :param row: DataFrame row
        :param method: replacement method ("dataset" or "Faker")
        :return: updated row
        """
        tag_name = [k for k, v in self.entity_functions.items() if v == self.replace_person][0]
        person_names = [(ind, row["words_pseudonymized"][ind], entity.split('-')[0]) for ind, entity in
                        enumerate(row["labels_pseudonymized"]) if entity.split('-')[-1] == tag_name]
        clean_names = self.concat_entities(person_names)

        replaced_words = row["words_pseudonymized"].copy()
        labels = row["labels_pseudonymized"].copy()
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
                replacement_name = self.match_punctuation(replacement_name, _name)
            replaced_words, labels = self.switch_entity(replaced_words, labels, _name, replacement_name, tag_name)

        return replaced_words, labels

    def replace_organization(self, row: pd.Series):
        """
        Replaces company names by fake ones.
        :param row: DataFrame row
        :return: updated row
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

        tag_name = [k for k, v in self.entity_functions.items() if v == self.replace_organization][0]
        all_lists = [gafam, banks, malls, security, internet, telecom]
        found_org = [(ind, row["words_pseudonymized"][ind], entity.split('-')[0]) for ind, entity in
                     enumerate(row["labels_pseudonymized"]) if entity.split('-')[-1] == tag_name]
        clean_org = self.concat_entities(found_org)

        replaced_words = row["words_pseudonymized"].copy()
        labels = row["labels_pseudonymized"].copy()
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
                    replacement_org = self.match_punctuation(replacement_org, org)
                # Modification du texte
                replaced_words, labels = self.switch_entity(replaced_words, labels, org, replacement_org,
                                                            tag_name)
        return replaced_words, labels

    def replace_loc(self, row: pd.Series):
        """
        Replaces locations by fake ones.
        :param row: DataFrame row
        :return: updated row
        """

        tag_name = [k for k, v in self.entity_functions.items() if v == self.replace_loc][0]
        _places = ["de la place", "du supermarché", "du centre commercial", "de l'hopital", "du cinéma", "de la mairie",
                   "du centre", "du marché", "de la bilbiothèque", "du magasin", "du parc"]
        found_loc = [(ind, row["words_pseudonymized"][ind], entity.split('-')[0]) for ind, entity in
                     enumerate(row["labels_pseudonymized"]) if entity.split('-')[-1] == tag_name]
        clean_loc = self.concat_entities(found_loc)

        replaced_words = row["words_pseudonymized"].copy()
        labels = row["labels_pseudonymized"].copy()
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
            replacement_loc = self.match_punctuation(replacement_loc, _loc)
            replaced_words, labels = self.switch_entity(replaced_words, labels, _loc, replacement_loc, tag_name)

        return replaced_words, labels

    def replace_date(self, row: pd.Series):
        """
        Replaces dates by fake ones.
        :param row: DataFrame row
        :return: updated row
        """

        tag_name = [k for k, v in self.entity_functions.items() if v == self.replace_date][0]
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

        found_dates = [(ind, row["words_pseudonymized"][ind], entity.split('-')[0]) for ind, entity in
                       enumerate(row["labels_pseudonymized"]) if entity.split('-')[-1] == tag_name]
        clean_dates = self.concat_entities(found_dates)

        replaced_words = row["words_pseudonymized"].copy()
        labels = row["labels_pseudonymized"].copy()
        if len(found_dates) > 0:
            for _date in clean_dates:
                # month only
                if unidecode.unidecode(_date.split(' ')[0]) in month_list:
                    replacement_date = datetime.datetime.now() - datetime.timedelta(days=randint(0, 365))
                    replacement_date = replacement_date.strftime("%b %Y")
                    replacement_date = self.match_case(replacement_date, _date)
                    replacement_date = self.match_punctuation(replacement_date, _date)
                elif len(_date) == 4 and all([x.isnumeric() for x in _date]):
                    replacement_date = str(randint(1900, datetime.datetime.now().year))
                    replacement_date = self.match_punctuation(replacement_date, _date)
                else:
                    replacement_date = datetime.datetime.now() - datetime.timedelta(days=randint(0, 365))
                    replacement_date = replacement_date.strftime(np.random.choice(date_formats_without_days, p=weights))
                    replacement_date = self.match_case(replacement_date, _date)
                    replacement_date = self.match_punctuation(replacement_date, _date)
                replaced_words, labels = self.switch_entity(replaced_words, labels, _date, replacement_date, tag_name)

        return replaced_words, labels

    def replace_time(self, row: pd.Series):
        """
        Replaces hours by fake ones.
        :param row: DataFrame row
        :return: updated row
        """

        tag_name = [k for k, v in self.entity_functions.items() if v == self.replace_time][0]
        locale.setlocale(locale.LC_ALL, 'fr_FR.utf-8')
        time_formats = ["%Hh%M", "%HH%M", "%-Hh%-M", "%-HH%-M", "%H:%M", "%H heures %M", "%-H heures %-M", "%H h %M",
                        "%H H %M",
                        "%-H h %-M", "%-H H %-M"]
        weights = [1, 0.1, 0.1, 0.05, 0.05, 0.3, 0.05, 0.02, 0.02, 0.02, 0.02]
        weights = [x / sum(weights) for x in weights]

        found_times = [(ind, row["words_pseudonymized"][ind], entity.split('-')[0]) for ind, entity in
                       enumerate(row["labels_pseudonymized"]) if entity.split('-')[-1] == tag_name]
        clean_times = self.concat_entities(found_times)

        replaced_words = row["words_pseudonymized"].copy()
        labels = row["labels_pseudonymized"].copy()
        if len(clean_times) > 0:
            former_text = ' '.join(row["words_pseudonymized"])
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
                    replacement_time = datetime.datetime.now().replace(hour=randint(0, 23), minute=randint(0, 59))
                    _previous_time = replacement_time
                    _previous_format = np.random.choice(time_formats, p=weights)
                else:
                    replacement_time = _previous_time + datetime.timedelta(minutes=randint(0, 180))
                replacement_time = replacement_time.strftime(_previous_format)
                replacement_time = self.match_case(replacement_time, _time)
                replacement_time = self.match_punctuation(replacement_time, _time)
                replaced_words, labels = self.switch_entity(replaced_words, labels, _time, replacement_time, tag_name)

        return replaced_words, labels

    def replace_address(self, row: pd.Series):
        """
        Replaces addresses by fake ones.
        :param row: DataFrame row
        :return: updated row
        """

        tag_name = [k for k, v in self.entity_functions.items() if v == self.replace_address][0]
        found_addresses = [(ind, row["words_pseudonymized"][ind], entity.split('-')[0]) for ind, entity in
                           enumerate(row["labels_pseudonymized"]) if entity.split('-')[-1] == tag_name]

        # If there is at least one person name, replace a random one. Else return None.
        clean_addresses = self.concat_entities(found_addresses)

        replaced_words = row["words_pseudonymized"].copy()
        labels = row["labels_pseudonymized"].copy()
        for _address in clean_addresses:
            with_num = sum(c.isdigit() for c in _address)
            if with_num > 0:
                _random_place = self.address.sample(1).iloc[0]
                replacement_address = "{} {}".format(randint(1, 60), _random_place.street)
            else:
                ind = min(find_sub_list(_address.split(' '), row["words_pseudonymized"])) - 1
                word_before = row["words_pseudonymized"][ind] if ind > 0 else None
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

            replacement_address = self.match_case(replacement_address, _address)
            replacement_address = self.match_punctuation(replacement_address, _address)
            replaced_words, labels = self.switch_entity(replaced_words, labels, _address, replacement_address,
                                                        tag_name)

        return replaced_words, labels

    def replace_cities(self, row: pd.Series):
        """
        Replaces city names by fake ones.
        :param row: DataFrame row
        :return: updated row
        """

        tag_name = [k for k, v in self.entity_functions.items() if v == self.replace_cities][0]
        found_cities = [(ind, row["words_pseudonymized"][ind], entity.split('-')[0]) for ind, entity in
                        enumerate(row["labels_pseudonymized"]) if entity.split('-')[-1] == tag_name]

        # If there is at least one person name, replace a random one. Else return None.
        clean_cities = self.concat_entities(found_cities)

        replaced_words = row["words_pseudonymized"].copy()
        labels = row["labels_pseudonymized"].copy()

        cities_dict = {}
        for _city in clean_cities:
            # If we already met this entity, we use the same substitution again
            if _city in cities_dict.keys():
                replacement_city = cities_dict[_city]
            # If this is the first time we meet this entity, we create a substitution
            else:
                _random_place = self.address.sample(1).iloc[0]
                replacement_city = _random_place.city
                replacement_city = self.match_case(replacement_city, _city)
                replacement_city = self.match_punctuation(replacement_city, _city)
                cities_dict[_city] = replacement_city
            replaced_words, labels = self.switch_entity(replaced_words, labels, _city, replacement_city, tag_name)

        return replaced_words, labels

    def replace_zip(self, row: pd.Series):
        """
        Replaces zip codes by fake ones.
        :param row: DataFrame row
        :return: updated row
        """

        tag_name = [k for k, v in self.entity_functions.items() if v == self.replace_zip][0]
        found_zip = [(ind, row["words_pseudonymized"][ind], entity.split('-')[0]) for ind, entity in
                     enumerate(row["labels_pseudonymized"]) if entity.split('-')[-1] == tag_name]

        # If there is at least one person name, replace a random one. Else return None.
        clean_zip = self.concat_entities(found_zip)
        clean_zip = [str(x) for x in clean_zip]

        replaced_words = row["words_pseudonymized"].copy()
        labels = row["labels_pseudonymized"].copy()
        for _zip in clean_zip:
            _random_place = self.address.sample(1).iloc[0]
            replacement_zip = str(_random_place.postcode)
            replaced_words, labels = self.switch_entity(replaced_words, labels, _zip, replacement_zip, tag_name)

        return replaced_words, labels

    def replace_email(self, row: pd.Series):
        """
        Replaces e-mail addresses by fake ones.
        :param row: DataFrame row
        :return: updated row
        """

        tag_name = [k for k, v in self.entity_functions.items() if v == self.replace_email][0]
        generators = [self.fake.ascii_company_email, self.fake.ascii_email, self.fake.ascii_free_email,
                      self.fake.ascii_safe_email]
        weights = [0.4, 0.5, 1, 0.1]
        weights = [x / sum(weights) for x in weights]

        found_mails = [(ind, row["words_pseudonymized"][ind], entity.split('-')[0]) for ind, entity in
                       enumerate(row["labels_pseudonymized"]) if entity.split('-')[-1] == tag_name]
        clean_emails = self.concat_entities(found_mails)

        replaced_words = row["words_pseudonymized"].copy()
        labels = row["labels_pseudonymized"].copy()
        mail_dic = {}
        for _mail in clean_emails:
            # If we already met this entity, we use the same substitution again
            if _mail in mail_dic.keys():
                replacement_mail = mail_dic[_mail]
            # If this is the first time we meet this entity, we create a substitution
            else:
                replacement_mail = np.random.choice(generators, p=weights)()
                replacement_mail = self.match_case(replacement_mail, _mail)
                replacement_mail = self.match_punctuation(replacement_mail, _mail)
                mail_dic[_mail] = replacement_mail
            replaced_words, labels = self.switch_entity(replaced_words, labels, _mail, replacement_mail, tag_name)

        return replaced_words, labels

    def replace_phone(self, row: pd.Series):
        """
        Replaces phone numbers by fake ones.
        :param row: DataFrame row
        :return: updated row
        """

        tag_name = [k for k, v in self.entity_functions.items() if v == self.replace_phone][0]
        found_phones = [(ind, row["words_pseudonymized"][ind], entity.split('-')[0]) for ind, entity in
                        enumerate(row["labels_pseudonymized"]) if entity.split('-')[-1] == tag_name]
        clean_phones = self.concat_entities(found_phones)

        replaced_words = row["words_pseudonymized"].copy()
        labels = row["labels_pseudonymized"].copy()
        for _phone in clean_phones:
            replacement_phone = self.fake.phone_number()
            replacement_phone = self.match_case(replacement_phone, _phone)
            replacement_phone = self.match_punctuation(replacement_phone, _phone)
            replaced_words, labels = self.switch_entity(replaced_words, labels, _phone, replacement_phone, tag_name)

        return replaced_words, labels

    def replace_immat(self, row: pd.Series):
        """
        Replaces car registration numbers by fake ones.
        :param row: DataFrame row
        :return: updated row
        """

        tag_name = [k for k, v in self.entity_functions.items() if v == self.replace_immat][0]
        links = ['-', '.', '', ' ']
        weights = [1, 0.1, 0.1, 0.1]
        weights = [x / sum(weights) for x in weights]

        found_immat = [(ind, row["words_pseudonymized"][ind], entity.split('-')[0]) for ind, entity in
                       enumerate(row["labels_pseudonymized"]) if entity.split('-')[-1] == tag_name]
        clean_immat = self.concat_entities(found_immat)

        replaced_words = row["words_pseudonymized"].copy()
        labels = row["labels_pseudonymized"].copy()
        for _immat in clean_immat:
            _link = np.random.choice(links, p=weights)
            replacement_immat = [choice(string.ascii_uppercase) for _ in range(2)] + [_link] + [choice(string.digits)
                                                                                                for _ in range(3)] + [
                                    _link] + [choice(string.ascii_uppercase) for _ in range(2)]
            replacement_immat = ''.join(replacement_immat)
            replaced_words, labels = self.switch_entity(replaced_words, labels, _immat, replacement_immat, tag_name)

        return replaced_words, labels

    def replace_money(self, row: pd.Series):
        """
        Replaces money amounts by fake ones.
        :param row: DataFrame row
        :return: updated row
        """

        tag_name = [k for k, v in self.entity_functions.items() if v == self.replace_money][0]
        currency = [' €', '€', ' euros', 'euros', ' Euros']
        weights = [1, 0.5, 0.5, 0.01, 0.02]
        weights = [x / sum(weights) for x in weights]

        found_money = [(ind, row["words_pseudonymized"][ind], entity.split('-')[0]) for ind, entity in
                       enumerate(row["labels_pseudonymized"]) if entity.split('-')[-1] == tag_name]
        clean_money = self.concat_entities(found_money)

        replaced_words = row["words_pseudonymized"].copy()
        labels = row["labels_pseudonymized"].copy()
        for _money in clean_money:
            alpha = random()
            if alpha < 0.5:
                _amount = "{0:.2f}".format(abs(np.random.normal(20, 100)))
            else:
                _amount = str(int(abs(np.random.normal(20, 100))))
            _currency = np.random.choice(currency, p=weights)
            replacement_money = _amount + np.random.choice(currency, p=weights)
            replacement_money = ''.join(replacement_money)
            replacement_money = self.match_punctuation(replacement_money, _money)
            replaced_words, labels = self.switch_entity(replaced_words, labels, _money, replacement_money, tag_name)

        return replaced_words, labels

    def replace_website(self, row: pd.Series):
        """
        Replaces websites by fake ones.
        :param row: DataFrame row
        :return: updated row
        """

        tag_name = [k for k, v in self.entity_functions.items() if v == self.replace_website][0]
        websites = ['Amazon', 'Facebook', 'Google', 'OVH', 'Orange', 'Paypal', 'Le bon coin', "Bon Coin", "Boncoin",
                    'Leboncoin', 'Microsoft', 'Instagram', 'Booking.com', 'Airbnb']
        domain_name = r"(\.[a-zA-Z]{2,3}\Z)"
        weights = [1 for _ in websites]
        weights = [x / sum(weights) for x in weights]

        found_website = [(ind, row["words_pseudonymized"][ind], entity.split('-')[0]) for ind, entity in
                         enumerate(row["labels_pseudonymized"]) if entity.split('-')[-1] == tag_name]
        clean_website = self.concat_entities(found_website)

        replaced_words = row["words_pseudonymized"].copy()
        labels = row["labels_pseudonymized"].copy()
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
                replacement_web = self.match_punctuation(replacement_web, _website)
                web_dict[_website] = replacement_web
            replaced_words, labels = self.switch_entity(replaced_words, labels, _website, replacement_web, tag_name)

        return replaced_words, labels

    def replace_car(self, row: pd.Series):
        """
        Replaces car models by fake ones.
        :param row: DataFrame row
        :return: updated row
        """

        tag_name = [k for k, v in self.entity_functions.items() if v == self.replace_car][0]
        colors_f = ['grise', 'blanche', 'noire', 'rouge', 'bleue', 'verte', 'beige']
        colors_h = ['gris', 'blanc', 'noir', 'rouge', 'bleu', 'vert', 'beige']

        found_car = [(ind, row["words_pseudonymized"][ind], entity.split('-')[0]) for ind, entity in
                     enumerate(row["labels_pseudonymized"]) if entity.split('-')[-1] == tag_name]
        clean_car = self.concat_entities(found_car)

        replaced_words = row["words_pseudonymized"].copy()
        labels = row["labels_pseudonymized"].copy()
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
                    ind = find_sub_list(_car.split(' '), row["words_pseudonymized"])[0] - 1
                except (Exception, ):
                    print(f"entity of type {_car} not found")
                if row["words_pseudonymized"][ind] in ['un', 'mon', 'son', 'le']:
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
            replacement_car = self.match_punctuation(replacement_car, _car)
            replaced_words, labels = self.switch_entity(replaced_words, labels, _car, replacement_car, tag_name)

        return replaced_words, labels

    def replace_refnum(self, row: pd.Series):
        """
        Replaces reference numbers by fake ones.
        :param row: DataFrame row
        :return: updated row
        """

        tag_name = [k for k, v in self.entity_functions.items() if v == self.replace_refnum][0]
        categories = ['rapport', 'contravention', 'constatation', 'commande', 'facture', 'courante', 'numero', 'iban',
                      'reference', 'bancaire', 'verbal', 'adresse', 'contrat', 'suivi', 'carte', 'cheque', 'code',
                      'colis', 'compte', 'dossier', 'puce', 'siret', 'siren', 'proces', 'ipv6', 'avis',
                      'produit', 'serie', 'bic', 'imei', 'ip', 'main', 'n°', 'ref', 'de', 'vol', ':']
        found_ref = [(ind, row["words_pseudonymized"][ind], entity.split('-')[0]) for ind, entity in
                     enumerate(row["labels_pseudonymized"]) if entity.split('-')[-1] == tag_name]
        clean_ref = self.concat_entities(found_ref)

        replaced_words = row["words_pseudonymized"].copy()
        labels = row["labels_pseudonymized"].copy()
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
            replacement_ref = self.match_punctuation(replacement_ref, _ref)
            replaced_words, labels = self.switch_entity(replaced_words, labels, _ref, replacement_ref, tag_name)

        return replaced_words, labels

    def switch_entity(self, words: List[str], labels: List[str], old_entity: str, new_entity: str, entity_type: str):
        """
        Replaces an entity by another in a text.
        :param words: text words
        :param labels: text labels
        :param old_entity: entity to replace
        :param new_entity: fake entity to insert
        :param entity_type: type of the entity to replace (without B- or I-)
        :return: updated words and labels
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

        new_entity = new_entity.replace(" ", self.white_space_token)
        if old_entity.endswith(self.white_space_token):
            new_entity += self.white_space_token
        found_right_entity = False
        char_start = 0
        max_pos = len(labels) - [x.split('-')[-1] for x in labels][::-1].index(entity_type) - 1
        i_s, i_e = [None] * 2
        while not found_right_entity:
            i_s, i_e = _find_sub_list(old_entity.split(' '), words, char_start)
            found_label = None if labels[i_s] == 'O' else labels[i_s].split("-")[-1]

            # we check whether we could find the entity in the text and if the corresponding labels are the right ones.
            # This prevents replacing a word or a sequence contained inside other entities.
            if (i_s is not None) & (found_label == entity_type):
                found_right_entity = True
            else:
                char_start = i_e + 1
                if char_start > max_pos:
                    break
        assert found_right_entity, "the identity we want to replace was not found"

        new_words = words[:i_s] + new_entity.split(' ') + words[i_e + 1:]
        entity_labels = ['I-' + entity_type for _ in new_entity.split(' ')]
        if len(entity_labels) > 1:
            entity_labels[0] = 'B-' + entity_type
        labels = labels[:i_s] + entity_labels + labels[i_e + 1:]

        return new_words, labels
