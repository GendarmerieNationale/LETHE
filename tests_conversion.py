import os
import json
from objects import Corpus


def convert_annotations(_input_path: str, _input_type: str, _output_path: str, _output_type: str):
    """
    Converts annotations between differents formats.
    :param _input_path: path of the annotations to convert
    :param _input_type: type of the initial annotations ("brat", "doccano" or "labelstudio")
    :param _output_path: converted annotations path
    :param _output_type: type of annotations after conversion ("brat", "doccano" or "labelstudio")
    """

    assert (_input_type in ["brat", "doccano", "labelstudio"]) & (_output_type in ["brat", "doccano", "labelstudio"]), \
        "Parameters input_type and output_type can only take the following values: 'brat', 'doccano' and 'labelstudio'."
    cc = Corpus()
    if _input_type == "brat":
        cc.load_brat_annotations(_input_path, "entity_name", "train")
    elif _input_type == "doccano":
        cc.load_doccano_annotations(_input_path, "entity_name", "train")
    elif _input_type == "labelstudio":
        cc.load_labelstudio_annotations(_input_path, "entity_name", "train")

    if _output_type == "brat":
        if not os.path.isdir(_output_path):
            os.mkdir(_output_path)
        annotations = cc.export_to_brat_annotations("train")
        for file in annotations:
            with open(os.path.join(_output_path, f"{file['id']}.txt"), "w") as outfile:
                outfile.write(file["txt"])
            with open(os.path.join(_output_path, f"{file['id']}.ann"), "w") as outfile:
                for line in file["ann"]:
                    outfile.write(line)
                    outfile.write("\n")
    elif _output_type == "doccano":
        annotations = cc.export_to_doccano_annotations("train")
        with open(_output_path, 'w') as outfile:
            for entry in annotations:
                json.dump(entry, outfile)
                outfile.write('\n')
    elif _output_type == "labelstudio":
        annotations = cc.export_to_labelstudio_annotations("train")
        with open(_output_path, "w") as outfile:
            outfile.write(json.dumps(annotations))


if __name__ == "__main__":

    input_path = 'data/samples/Doccano/sample.jsonl'
    input_type = "doccano"
    output_path = 'data/samples/label-studio/converted_sample.jsonl'
    output_type = "labelstudio"
    convert_annotations(input_path, input_type, output_path, output_type)
