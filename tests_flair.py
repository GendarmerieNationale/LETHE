import os
from typing import List
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, StackedEmbeddings, TokenEmbeddings, WordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.optim import LinearSchedulerWithWarmup
from torch.optim import AdamW
from torch.cuda import is_available
from objects import Corpus, EntityDetector, Pseudo


def flair_predict_sentence(text_for_ner, _model_path):

    _tagger = SequenceTagger.load(_model_path)
    sentence = Sentence(text_for_ner)
    _tagger.predict(sentence)
    entities = sentence.get_spans('ner')
    return entities


def flair_model_training(_corpus: Corpus, _corpus_keys: dict, _embedding_types: List[TokenEmbeddings], _hyperparams,
                         scheduler=None, output_folder="models/trained_flair"):

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    _corpus = _corpus.export_to_flair_dataset({"train": _corpus_keys["train"], "test": _corpus_keys["test"],
                                               "dev": _corpus_keys["dev"]})
    tag_dictionary = _corpus.make_label_dictionary(label_type="ner")

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=_embedding_types)

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256, embeddings=embeddings, use_crf=True, tag_dictionary=tag_dictionary, tag_type="ner"
    )

    trainer: ModelTrainer = ModelTrainer(model=tagger, corpus=_corpus)

    t_total = len(_corpus.train) // _hyperparams["gradient_accumulation_steps"] * _hyperparams["num_train_epochs"]

    # Here, we only train the last layers of the networks
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in tagger.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": _hyperparams["weight_decay"], "lr": _hyperparams["learning_rate"],
         "eps": _hyperparams["adam_epsilon"]},
        {"params": [p for n, p in tagger.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,
         "lr": _hyperparams["learning_rate"], "eps": _hyperparams["adam_epsilon"]}
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    if scheduler is None:
        scheduler = LinearSchedulerWithWarmup(optimizer, num_warmup_steps=_hyperparams["warmup_steps"],
                                              num_train_steps=t_total)

    trainer.train(
        output_folder,
        max_epochs=_hyperparams["num_train_epochs"],
        learning_rate=_hyperparams["learning_rate"],
        mini_batch_size=_hyperparams["mini_batch_size"],
        embeddings_storage_mode="cuda" if is_available() else "cpu",
        checkpoint=False,
        scheduler=scheduler
    )


if __name__ == "__main__":

    # Chargement de corpus annotés ou non et entrainement d'un modèle flair
    corpus = Corpus()

    corpus.load_brat_annotations('data/samples/BRAT', "entity_name", "brat")
    corpus.load_doccano_annotations('data/samples/Doccano/sample.jsonl', "entity_name", "doccano")
    corpus.load_labelstudio_annotations('data/samples/label-studio/sample.json', "entity_name", "labelstudio")
    text = """C’est une épave mythique 10h57. Celle de l’Endurance, le navire de l’explorateur britannique Ernest Shackleton, brisé par les glaces en 1915 au large de l’Antarctique. Elle a été retrouvée dans la mer de Weddell par 3 000 mètres de fond, ont annoncé ses découvreurs, mercredi 9 mars."""
    corpus.load_list_of_texts([text], "list")
    # corpus.filter_labels(["PERSON"])

    # Utilisation d'un modèle de détection des entités nommées
    model_path = "models/ner-french/pytorch_model.bin"
    model_labels = ["O", "PER", "MISC", "ORG", "LOC"]
    anonymizer = EntityDetector("flair", model_path, model_labels)
    anonymizer.process_corpus_flair(corpus, "brat")
    anonymizer.process_corpus_regex(corpus, "brat")

    # Pseudonymisation du corpus
    pseudonymizer = Pseudo("data/datasets/noms_prenoms.csv", "data/datasets/adresses_sample.csv",
                           "data/datasets/cars_sample.csv", "data/datasets/societes.csv", "tags", " ")
    pseudonymizer.replace_entities_by_fake_ones(corpus, "doccano", ["CITY", "DATE", "PERSON"], "words_labels")

    # Entrainement d'un modèle NER
    embedding_types = [
        FlairEmbeddings("fr-forward"),
        FlairEmbeddings("fr-backward"),
        WordEmbeddings("fr")
    ]
    hyperparams = {"adam_epsilon": 1e-8, "warmup_steps": 10, "num_train_epochs": 2, "gradient_accumulation_steps": 1,
                   "weight_decay": 0.0, "learning_rate": 0.1, "mini_batch_size": 32}
    output_model_folder = "data/test/"
    corpus_keys = {"train": "brat", "test": "doccano", "dev": "labelstudio"}
    flair_model_training(corpus, corpus_keys, embedding_types, hyperparams, output_model_folder)
