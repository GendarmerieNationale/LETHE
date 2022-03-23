<div id="top"></div>

<div align="center">
  <a href="https://omnibus-pic.gendarmerie.fr/DATALAB/projets/pseudonymisation">
    <img src="logo_Datalab.png" alt="Logo" width="96" height="80">
  </a>
</div>

# LÉTHÉ
(Librairy for Extraction of Textual Hidden Entities)

<details>
  <summary>Sommaire</summary>
  <ol>
    <li><a href="#description">Description du projet</a></li>
    <li>
      <a href="#installation">Installation </a>
      <ul>
        <li><a href="#clonage_repo">Clonage du dépôt</a></li>
        <li><a href="#virtual_env">Création de l'environement virtuel</a></li>
        <li><a href="#python_packages">Installation des paquets python</a></li>
      </ul>
    </li>
    <li><a href="#python_files">Fichiers python</a></li>
    <li><a href="#json_files">Fichiers de configuration</a></li>
    <li><a href="#corpus">Corpus de textes</a></li>
    <li><a href="#datasets">Datasets de remplacement</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


## I. Description du projet
<div id="description"></div>

Le code de ce projet a pour but de permettre la pseudonymisation automatique d'un corpus de textes.
Concrètement, le processus s'effectue en 2 temps:
- un modèle de deep learning détecte un certain nombre d'entités nommées.
- ces entités sont remplacées dans le corpus par une balise indiquant leur type ou une entité fictive.

Le projet inclut des fonctions permettant de :
- charger un corpus de textes annoté au format doccano (jsonl)
- charger un corpus de textes non annotés sous forme de liste de chaines de caractères (["blabla", "blablabla",...])
- détecter une liste d'entités dans un corpus de textes
- remplacer les entités annotées par une balise ou d'autres entités fictives du meme type
- entrainer un modèle de type Transformers sur un corpus de textes annotés
- évaluer les performances de détection d'entités nommées d'un modèle sur un corpus test
- exporter un corpus de texte au format doccano

<p align="right">(<a href="#top">retour en haut</a>)</p>

## II. Installation
<div id="installation"></div>

1. Clonage du dépôt
   <div id="clonage_repo"></div>
   
   ```sh
   git clone https://omnibus-pic.gendarmerie.fr/DATALAB/projets/jud/automis.git
   ```
2. Création de l'environement virtuel
   <div id="virtual_env"></div>
   
    Dans le dossier du projet,
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Installation des paquets python
   <div id="python_packages"></div>
   
   ```sh
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#top">retour en haut</a>)</p>


## III. Fichiers python
<div id="python_files"></div>

#### Fichier `pseudo.py`

Ce fichier contient les classes permettant de faire de la reconnaissance d'entités nommées (entrainement de modèles,
évaluation, annotation de corpus) et de remplacer des entités annotées par une balise ou une entité fictive du meme type.
Il contient également les fonctions permettant d'effectuer l'ensemble des taches détaillées dans lapartie précédente de A à Z.

#### Fichier `utils.py`

Ensemble de fonctions essentiellement liées au processus de tokenization du corpus de textes.

#### Fichier `tests.py` 

Exemples d'appels aux fonctions de base.

#### Fichier `focal_loss.py`

Implémentation de la fonction focal loss pouvant etre appelée pour l'entrainement de modèles.

<p align="right">(<a href="#top">retour en haut</a>)</p>

## IV. Fichiers de configuration
<div id="json_files"></div>

Un fichier de configuration au format json doit etre fournie à l'algorithme. Son format dépend de la tache à effectuer. En voici un exemple:

```json
{
  "output_dir": "data/test/",
  "model_path": "data/modele/FlaubertForTokenClassification.pt",
  "tokenizer_path": "data/tokenizer",
  "labels_format": "entity_name",
  "seed": 42,
  "max_seq_length": 128,
  "per_gpu_batch_size": 8,
  "adam_epsilon": 1e-8,
  "gradient_accumulation_steps": 1,
  "max_grad_norm": 1.0,
  "max_steps": -1,
  "num_train_epochs": 2,
  "weight_decay": 0.0,
  "learning_rate": 5e-5,
  "warmup_steps": 0,
  "loss_function": "CrossEntropy",
  "entities": [
    "PERSON",
    "DATE",
    "GPE",
    "IMMAT",
    "TIME",
    "ADDRESS",
    "WEBSITE",
    "MONEY",
    "CAR",
    "REF_NUM",
    "EMAIL",
    "ZIP",
    "PHONE"
],
  "regex": []
}
```
Comme indiqué dans l'exemple ci-dessus, le champ `model_path` doit etre un fichier `.pt` correspondant à un modèle de 
type Transformers. Le champ `tokenizer_path` doit etre un dossier contenant un dossier du nom du modèle indiqué dans 
`model_path` ("FlaubertForTokenClassification" dans le cas précédent). Celui-ci doit contenir les fichiers suivants:
- config.json
- tokenizer_config.json
- vocab.json
- merges.txt

Le dossier `data` contient les schémas des fichiers json attendus selon la tache que l'on souhaite effectuer ainsi que des
exemples de chaque type.

<p align="right">(<a href="#top">retour en haut</a>)</p>

## V. Corpus de textes
<div id="corpus"></div>

Les corpus de textes peuvent etre fournis directement sous forme d'une liste de chaines de caractères en appelant la 
fonction `load_custom_corpus` ou bien sous la forme d'un fichier `.jsonl` au format doccano en appelant la fonction 
`load_doccano_corpus` (voir `test.py`). Le dossier `samples` contient 2 exemples de corpus doccano.

<p align="right">(<a href="#top">retour en haut</a>)</p>


## V. Datasets de remplacement
<div id="datasets"></div>

Les entités nommées détectées peuvent etre remplacées par d'autres entités fictives du meme type. Certaines d'entre 
elles sont générées en piochant de manière aléatoire dans des jeux de données. C'est le cas des adresses, des modèles de
voitures, des noms propres et des noms de sociétés. Ces jeux de données ne sont pas inclus dans le repo git et devront
etre ajoutés par l'utilisateur qui en précisera le chemin (voir fichier `tests.py`). Les fichiers utilisés initialement
ont le format suivant (les colonnes au noms barrés ne sont pas utilisées et donc facultatives):

### Adresses

| street | postcode | city | street_type |
|-----------|:-----------:|-----------:| -----------:|
|Résidence Anatole France|57000|Metz|Résidence|
|Rue de Courtelevant|19160|Liginiac|Rue|
|Rue du Vallon (Annecy-le-Vieux)|88290|Saulxures-sur-Moselotte|Rue|
|Cite Charcot Spanel|17110|Saint-Georges-de-Didonne|Cite|
| ... | | | |


### Modèles de voitures

| marque | modele | ~~nombre~~ | genre |
|-----------|:-----------:|-----------:| -----------:|
| Peugeot | 2008 | 150498 | F |
| Citroen | Berlingo | 125393 | F |
| Peugeot | Partner | 124670 | F |
| Renault | Megane | 115236 | F |
| ... | | | |


### Noms propres

| noms | noms_poids | prenoms | prenoms_poids |
|-----------|:-----------:|-----------:| -----------:|
| Bort | 1.8636550001863656E-5 | Jocelin | 1.1957508991797646E-5 |
| Chevron | 2.8231605448367715E-5 | Suzane | 9.181658690130337E-6 |
| Christien | 3.247557228047528E-5 | Tewfik | 8.825780446326835E-6 |
| Oberti | 2.251147623987491E-5 |  | 0.0 |
| ... | | | |

Le colonnes terminant par `_poids` correspondent à la fréquence d'apparition des noms ou prénoms.


### Noms de sociétés

| Dénomination |
|-----------|
| PROTECH |
| MILANO |
| MANU VTC Transport-service |
| MARKETING COMMUNICATION PUBLICITE LTD (MCP) |
|...|


L'utilisateur pourra trouver des données similaires sur des sites connus d'open data.

<p align="right">(<a href="#top">retour en haut</a>)</p>

## VI. License
<div id="license"></div>

Distribué sous licence Apache-2.0. Voir `LICENSE.txt` pour plus d'informations.

<p align="right">(<a href="#top">retour en haut</a>)</p>

## VII. Contact
<div id="contact"></div>

STSI² / SDSI / BAP / DATALAB  - datalab.bap.stsisi@gendarmerie.interieur.gouv.fr

Lien du projet: [Pseudonymisation](https://omnibus-pic.gendarmerie.fr/DATALAB/projets/pseudonymisation)

<p align="right">(<a href="#top">retour en haut</a>)</p>
