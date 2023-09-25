from __future__ import absolute_import, division, print_function

import logging
import os
import torch
import pandas as pd
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
logger = logging.getLogger(__name__)


def build_tags(df):
    """
    Créé les tags (au format BIO) correspondant aux entités annotées.
    :param df: DataFrame du corpus avec au moins une colonne 'word' et une 'type'
    return: liste des entités, liste des tags correspondants au format BIO
    """
    if df is None:
        df = pd.DataFrame()

    words = []
    labels = []
    for ind, row in df.iterrows():
        # row looks like :    T1    ORGANIZATION     Hello Bank
        entity_parts = str(row.word).replace("\xa0", ' ').strip().split(" ")
        entity_parts = [x for x in entity_parts if len(x) > 0]
        entity_len = len(entity_parts)
        entity_type = row.type
        if entity_len > 1:
            _prefix = ["I-"] * entity_len
            _prefix[0] = "B-"
            _type = [entity_type] * entity_len
            tags = [x[0] + x[1] for x in zip(_prefix, _type)]
            words.append(entity_parts)
            labels.append(tags)
        else:
            words.append(entity_parts)
            labels.append(["I-" + row.type])
    return words, labels


def condense_labels(labels):
    new_labels = labels[:]
    for i in range(1, len(labels)):
        if (labels[i]["value"]["labels"] == labels[i - 1]["value"]["labels"]) & (
                labels[i]["value"]["start"] == labels[i - 1]["value"]["end"] + 1):
            new_labels.pop(i)
            new_labels[i-1]["value"]["end"] = labels[i]["value"]["end"]
            new_labels[i - 1]["value"]["text"] += f" {labels[i]['value']['text']}"
            new_labels = condense_labels(new_labels)
            break
    return new_labels


def find_sub_list(sublist, the_list, strict=True):
    """
    Trouve une liste dans une liste de listes et renvoie les indices de début et de fin de cette dernière.
    :param sublist: liste
    :param the_list: liste de listes
    :param strict: booléen valant True si l'on cherche l'élément exact ou False si l'on cherche des listes contenant l'
    élément
    :return: indices de début et de fin de la liste recherchée dans la liste de listes
    """

    sll = len(sublist)
    # If strict is set to True, we search for a succession of elements that strictly matches the sublist.
    if strict:
        for ind in (i for i, e in enumerate(the_list) if e == sublist[0]):
            if ' '.join(the_list[ind:ind + sll]) == ' '.join(sublist):
                return list(range(ind, ind + sll))
    # If strict is set to False, we search for a succession of elements that contains the sublist.
    else:
        ll = ['' if x is None else x for x in the_list]
        for ind in (i for i, e in enumerate(ll) if sublist[0] in e):
            if ' '.join(sublist) in ' '.join(the_list[ind:ind + sll]):
                return list(range(ind, ind + sll))


def get_text_and_labels(text, annotations, labels_format):
    """
    Convertit une chaine de caractères et ses annotations en listes de mots et de tags.
    :param text: texte (string)
    :param annotations: DataFrame des annotations associées au texte
    :param labels_format: format des annotations
    :return: liste de mots, liste des tags correspondants
    """

    text = split_text(text, sorted(list(zip(annotations['start_pos'].tolist(), annotations['end_pos'].tolist())),
                                   key=lambda x: x[0]))

    words, tags = build_tags(annotations)

    bio_tags = ['O' for _ in text]

    _text = text.copy()
    for _word, _label in zip(words, tags):
        ind = find_sub_list(_word, _text)
        if ind is None:
            print(f"Tagged word '{_word}' not found")
            continue
        for i, _tag in zip(ind, _label):
            bio_tags[i] = _tag
        _text = [None for _ in _text[:ind[0] + len(_word)]] + _text[min(len(_text), ind[0] + len(_word)):]

    if labels_format == "entity_name":
        bio_tags = [x.split("-")[-1] for x in bio_tags]

    return text, bio_tags


def extend_labels_to_tokens(label, n_subwords, method="full"):

    if method == "first":
        return [label] + [""] * (n_subwords-1)
    else:
        return [label] * (n_subwords)


def condense_labels_to_words(tokens, tokens_labels, first_token_symbol="▁", method="any"):

    if len(tokens) == 0:
        return []

    condensed_labels = []
    word_labels = [tokens_labels[0]]
    for token, label in list(zip(tokens, tokens_labels))[1:]:
        if token.startswith(first_token_symbol):
            condensed_labels.append(word_labels)
            word_labels = [label]
        else:
            word_labels += [label]
    condensed_labels.append(word_labels)
    if method == "any":
        condensed_labels = ["O" if all([y == "O" for y in x]) else max(set([y for y in x if y != "O"]),
                            key=[y for y in x if y != "O"].count) for x in condensed_labels]
    elif method == "first":
        condensed_labels = [x[0] for x in condensed_labels]

    return condensed_labels


def load_model_and_tokenizer(model_path, tokenizer_path, labels=None):
    """
    Chargement du modèle et du tokenizer associé.
    :param model_path: chemin vers le modèle à utiliser
    :param tokenizer_path: chemin vers le tokenizer à utiliser
    :param labels: liste des différents labels possibles
    :return: modèle et tokenizer
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_path.endswith(".pt"):
        if device.type == "cpu":
            model = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            model = torch.load(model_path)

        tokenizer_path = os.path.join(tokenizer_path, model.__class__.__name__)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        config_file = os.path.join(model_path, "config.json")

        config = AutoConfig.from_pretrained(config_file)
        if labels is not None:
            config.num_labels = len(labels)

        model = AutoModelForTokenClassification.from_config(config)

        tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def add_special_tokens_and_padding(text_tokens_ids, label_ids, file_tokens_ids, token_config, special_tokens,
                                   max_seq_length):

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    text_tokens_ids += [special_tokens["sep_id"]]
    label_ids += [special_tokens["pad_label_id"]]
    if token_config["extra_sep"]:
        # roberta uses an extra separator b/w pairs of sentences
        text_tokens_ids += [special_tokens["sep_id"]]
        label_ids += [special_tokens["pad_label_id"]]

    if token_config["cls_token_at_end"]:
        text_tokens_ids += [special_tokens["cls_id"]]
        label_ids += [special_tokens["pad_label_id"]]
    else:
        text_tokens_ids = [special_tokens["cls_id"]] + text_tokens_ids
        label_ids = [special_tokens["pad_label_id"]] + label_ids

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    mask_ids = [1 if token_config["mask_padding_with_zero"] else 0] * len(text_tokens_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(text_tokens_ids)
    if token_config["pad_on_left"]:
        text_tokens_ids = ([special_tokens["pad_id"]] * padding_length) + text_tokens_ids
        mask_ids = ([0 if token_config["mask_padding_with_zero"] else 1] * padding_length) + mask_ids
        label_ids = ([special_tokens["pad_label_id"]] * padding_length) + label_ids
    else:
        text_tokens_ids += ([special_tokens["pad_id"]] * padding_length)
        mask_ids += ([0 if token_config["mask_padding_with_zero"] else 1] * padding_length)
        label_ids += ([special_tokens["pad_label_id"]] * padding_length)

    padding_length = max_seq_length - len(file_tokens_ids)
    file_tokens_ids += ([special_tokens["pad_id"]] * padding_length)

    assert len(text_tokens_ids) == max_seq_length
    assert len(mask_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(file_tokens_ids) == max_seq_length

    return text_tokens_ids, mask_ids, label_ids, file_tokens_ids


def split_text(text, list_tuples):
    """
    Partage un texte en mots selon les espaces et en prenant en compte les annotations (utile si la fin d'une entité
    annotée ne correspond pas à un espace).
    :param text: texte (string)
    :param list_tuples: liste de tuples au format (caractère du début de l'entité annotée, caractère de fin de
    l'entité)
    :return: liste des mots du texte
    """

    prev_ind = 0
    _parts = []
    for i_s, i_e in list_tuples:
        _parts.append(text[prev_ind: i_s])
        _parts.append(text[i_s: i_e])
        prev_ind = i_e
    _parts.append(text[prev_ind:])
    # replacing non-breaking spaces
    _parts = [x.replace("\xa0", ' ') for x in _parts]
    _parts = [x.split(' ') for x in _parts]
    _parts = [item for sublist in _parts for item in sublist if len(item) > 0]
    return _parts


def split_into_parts(text_ids, labels_ids, file_ids, max_seq_length):
    """
    Coupe les textes trop longs en plusieurs parties.
    :param text_ids: ids des tokens du texte
    :param labels_ids: labels correspondants
    :param file_ids: ids des tokens du nom du fichier
    :param max_seq_length: taille maximale acceptée par le modèle (en nombre de tokens)
    :return: liste d'objets InputFeatures modifiée
    """
    _keep_cutting = True
    start_index = 0
    part_index = 1

    chunks = []
    while _keep_cutting:
        if len(text_ids[start_index:]) > max_seq_length:
            slice_text_ids = text_ids[start_index:max_seq_length]
            slice_label_ids = labels_ids[start_index:max_seq_length]
            start_index += max_seq_length
        else:
            slice_text_ids = text_ids[start_index:]
            slice_label_ids = labels_ids[start_index:]
            _keep_cutting = False

        chunks.append({"tokens_ids": slice_text_ids, "labels_ids": slice_label_ids, "tokens_ids_file": file_ids,
                       "part_index": part_index})
        part_index += 1

    return chunks


def switch_entity(x, old_entity, new_entity, entity_type, split_first=False):
    """
    Remplace une entité par une autre.
    :param x: ligne du DataFrame
    :param old_entity: entité à remplacer
    :param new_entity: entité de substitution
    :param entity_type: tag de l'entité (sans le préfixe B- ou I-)
    :param split_first: booléen indiquant si le texte doit etre partagé avant ou non
    :return: texte et tags modifiés
    """

    def _find_sub_list(sublist, the_list, start=0):
        """
        Finds a sublist in another list and returns the starting and ending indices.
        :param sublist: sublist
        :param the_list: list
        :param start : starting index (all elements before that will be ignored by the search)
        :return: starting and ending positions of the sublist in the list
        """
        sll = len(sublist)
        the_list = [None for _ in range(start)] + the_list[start:]
        for ind in (i for i, e in enumerate(the_list) if e == sublist[0]):
            if the_list[ind:ind + sll] == sublist:
                return ind, ind + sll - 1
        return None, None

    found_right_entity = False
    char_start = 0
    max_pos = len(x["labels"]) - [y.split('-')[-1] for y in x["labels"]][::-1].index(entity_type) - 1
    i_s, i_e = [None] * 2
    while not found_right_entity:
        if split_first:
            i_s, i_e = _find_sub_list(old_entity.split(' '), x["text"], char_start)
        else:
            i_s, i_e = _find_sub_list([old_entity], x["text"], char_start)

        found_label = None if x["labels"][i_s] == 'O' else x["labels"][i_s].split("-")[-1]

        # we check whether we could find the entity in the text and if the corresponding labels are the right ones.
        # This prevents replacing a word or a sequence contained inside other entities.
        if (i_s is not None) & (found_label == entity_type):
            found_right_entity = True
        else:
            char_start = i_e + 1
            if char_start > max_pos:
                break
    assert found_right_entity, "the identity we want to replace was not found"

    if split_first:
        x["text"] = x["text"][:i_s] + new_entity.split(' ') + x["text"][i_e + 1:]
    else:
        x["text"] = x["text"][:i_s] + [new_entity] + x["text"][i_e + 1:]

    if split_first:
        entity_labels = ['I-' + entity_type for _ in new_entity.split(' ')]
        if len(entity_labels) > 1:
            entity_labels[0] = 'B-' + entity_type
        x["labels"] = x["labels"][:i_s] + entity_labels + x["labels"][i_e + 1:]
    else:
        x["labels"] = x["labels"][:i_s] + [entity_type] + x["labels"][i_e + 1:]

    return x["text"], x["labels"]


def tokenize_and_preserve_labels(tokenizer, sentence, text_labels, file):
    """
    Convertit une phrase en tokens et crée une liste de labels correspondants.
    :param tokenizer: tokenizer du modèle
    :param sentence: phrase à tokenizer
    :param text_labels: labels de la phrase
    :param file: nom du texte
    :return: liste de tokens de la phrase, labels correspondant, liste de tokens du nom du texte
    """
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    # Tokenize file name so that we can store it into the container
    file_tokens = tokenizer.tokenize(str(file))

    return tokenized_sentence, labels, file_tokens


def tokenize_this(text_tokens, label_ids, tokenizer, max_seq_length, token_params, file_tokens):
    """

    :param text_tokens: tokens du texte
    :param label_ids: tags correspondants
    :param tokenizer: tokenizer du modèle
    :param max_seq_length: taille maximum de tokens prise par le modèle
    :param token_params: tokens spéciaux du tokenizer
    :param file_tokens: tokens du nom du texte
    :return: ids des tokens du texte, masque, , ids des tags, ids du nom du texte
    """
    sep_token, cls_token, pad_token, pad_token_label_id, pad_token_segment_id, sequence_a_segment_id, \
        cls_token_segment_id, sep_token_extra, cls_token_at_end, mask_padding_with_zero, pad_on_left = token_params

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    text_tokens += [sep_token]
    label_ids += [pad_token_label_id]
    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        text_tokens += [sep_token]
        label_ids += [pad_token_label_id]
    segment_ids = [sequence_a_segment_id] * len(text_tokens)

    if cls_token_at_end:
        text_tokens += [cls_token]
        label_ids += [pad_token_label_id]
        segment_ids += [cls_token_segment_id]
    else:
        text_tokens = [cls_token] + text_tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    file_token_ids = tokenizer.convert_tokens_to_ids(file_tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    file_padding_length = max_seq_length - len(file_token_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        label_ids = ([pad_token_label_id] * padding_length) + label_ids
        file_token_ids = ([pad_token] * file_padding_length) + file_token_ids
    else:
        input_ids += ([pad_token] * padding_length)
        input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids += ([pad_token_segment_id] * padding_length)
        label_ids += ([pad_token_label_id] * padding_length)
        file_token_ids += ([pad_token] * file_padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, label_ids, file_token_ids
