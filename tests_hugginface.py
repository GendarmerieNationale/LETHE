from utils import load_model_and_tokenizer
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import TrainingArguments, AutoTokenizer, AutoModelForTokenClassification, Trainer, pipeline
from flair.optim import LinearSchedulerWithWarmup
from objects import Corpus, Pseudo, EntityDetector


def hugginface_model_training(_corpus: Corpus, _corpus_keys: dict, _model, _tokenizer, _hyperparams, _model_columns,
                              scheduler=None, optimizer=None, output_folder="models/trained_hugginface"):

    tokenizer_config = {"extra_sep": False, "pad_on_left": False, "cls_token_at_end": False,
                        "mask_padding_with_zero": True}

    special_tokens = {"cls_id": _tokenizer.cls_token_id, "sep_id": _tokenizer.sep_token_id,
                      "pad_id": _tokenizer.pad_token_id, "cls_segment_id": 0,
                      "pad_segment_id": 0, "pad_label_id": CrossEntropyLoss().ignore_index,
                      "sequence_a_segment_id": 0}

    _corpus.tokenize_and_keep_labels(_tokenizer, _corpus_keys["train"], method="first")
    _corpus.tokenize_and_keep_labels(_tokenizer, _corpus_keys["test"], method="first")

    train_dataset = _corpus.export_to_hugginface_dataset(_hyperparams["input_vector_size"], tokenizer_config,
                                                         special_tokens, _corpus_keys["train"], _model_columns)
    eval_dataset = _corpus.export_to_hugginface_dataset(_hyperparams["input_vector_size"], tokenizer_config,
                                                        special_tokens, _corpus_keys["test"], _model_columns)

    training_args = TrainingArguments(output_dir=output_folder, evaluation_strategy="epoch")

    decay = ["classifier"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in _model.named_parameters() if any(nd in n for nd in decay)],
         "weight_decay": _hyperparams["weight_decay"], "lr": _hyperparams["learning_rate"],
         "eps": _hyperparams["adam_epsilon"]},
        {"params": [p for n, p in _model.named_parameters() if not any(nd in n for nd in decay)], "weight_decay": 0.0,
         "lr": _hyperparams["learning_rate"], "eps": _hyperparams["adam_epsilon"]}
    ]
    if optimizer is None:
        optimizer = AdamW(optimizer_grouped_parameters)
    if scheduler is None:
        scheduler = LinearSchedulerWithWarmup(optimizer, num_warmup_steps=_hyperparams["warmup_steps"],
                                              num_train_steps=_hyperparams["warmup_steps"])
    trainer = Trainer(
        model=_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, scheduler)
    )
    trainer.train()


if __name__ == "__main__":

    # Chargement d'un corpus annoté et entrainement d'un modèle hugginface
    corpus = Corpus()
    corpus.load_doccano_annotations('data/samples/Doccano/sample.jsonl', "entity_name", "doccano")
    corpus.load_brat_annotations('data/samples/BRAT', "entity_name", "brat")
    text = """C’est une épave mythique. Celle de l’Endurance, le navire de l’explorateur britannique Ernest Shackleton, brisé par les glaces en 1915 au large de l’Antarctique. Elle a été retrouvée dans la mer de Weddell par 3 000 mètres de fond, ont annoncé ses découvreurs, mercredi 9 mars."""
    corpus.load_list_of_texts([text], "list")
    # corpus.filter_labels(["PERSON", "GPE"])

    # Utilisation d'un modèle de détection des entités nommées
    model_path = "models/camembert-ner"
    tokenizer_config = {"extra_sep": False, "pad_on_left": False, "cls_token_at_end": False,
                        "mask_padding_with_zero": True}
    model_labels = ["O", "PER", "MISC", "ORG", "LOC"]
    pad_label_id = CrossEntropyLoss().ignore_index
    anonymizer = EntityDetector("hugginface", model_path, model_labels, pad_label_id, tokenizer_config)
    anonymizer.process_corpus_hugginface(corpus, "doccano")

    # Pseudonymisation du corpus
    pseudonymizer = Pseudo("data/datasets/noms_prenoms.csv", "data/datasets/adresses_sample.csv",
                           "data/datasets/cars_sample.csv", "data/datasets/societes.csv", "tags", " ")
    pseudonymizer.replace_entities_by_fake_ones(corpus, "doccano", ["ORG", "LOC", "PER"], "predicted_words_labels")

    # Loading tokenizer and define its configuration
    model, tokenizer = load_model_and_tokenizer(model_path, model_path, corpus.annotation_labels)
    model_columns = ["input_ids", "attention_mask", "labels"]
    hyperparams = {"adam_epsilon": 1e-8, "warmup_steps": 10, "num_train_epochs": 2, "gradient_accumulation_steps": 1,
                   "weight_decay": 0.01, "learning_rate": 5e-5, "mini_batch_size": 32, "input_vector_size": 128}
    corpus_keys = {"train": "brat", "test": "doccano", "dev": "labelstudio"}
    hugginface_model_training(corpus, corpus_keys, model, tokenizer, hyperparams, model_columns,
                              output_folder="models/trained_hugginface")
