import spacy
import json
import random
import logging
from spacy.scorer import Scorer
from sklearn.metrics import accuracy_score
from spacy.training.example import Example
import string
from pathlib import Path

output_dir = "/Users/Louis/Desktop/DataGlacier/dataglacier/week7"


def remove_whitespace_punctuation(text, start, end):
    # Remove leading and trailing whitespace and punctuation
    while start < end and text[start] in string.whitespace + string.punctuation:
        start += 1
    while start < end and text[end - 1] in string.whitespace + string.punctuation:
        end -= 1
    return start, end


def convert_dataturks_to_spacy(dataturks_JSON_FilePath):
    try:
        training_data = []
        lines = []
        with open(dataturks_JSON_FilePath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content'].replace('-', ' ')
            entities = []
            for annotation in data['annotation']:
                # only a single point in text annotation.
                point = annotation['points'][0]
                labels = annotation['label']
                # handle both list of labels or a single label.
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    # dataturks indices are both inclusive [start, end] but spacy is not [start, end)
                    label = label.replace('-', '_')
                    start, end = remove_whitespace_punctuation(
                        text, point['start'], point['end'] + 1)
                    entities.append((start, end, label))

            training_data.append((text, {"entities": entities}))

        return training_data
    except Exception as e:
        logging.exception("Unable to process " +
                          dataturks_JSON_FilePath + "\n" + "error = " + str(e))
        return None


################### Train Spacy NER.###########


def train_spacy():

    TRAIN_DATA = convert_dataturks_to_spacy(
        "traindata.json")

    TRAIN_DATA = clean_entities(TRAIN_DATA)
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        # ner = nlp.create_pipe('ner')
        # nlp.add_pipe(ner, last=True)

        ner = nlp.add_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(10):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                doc = nlp.make_doc(text)
                examples = [Example.from_dict(doc, annotations)]
                nlp.update(
                    examples,
                    drop=0.5,
                    sgd=optimizer,
                    losses=losses)
            print(losses)

    nlp.to_disk(Path(output_dir))
    print("Saved model to", output_dir)


def clean_entities(training_data):
    clean_data = []
    for text, annotation in training_data:

        entities = annotation.get('entities')
        entities_copy = entities.copy()

        # append entity only if it is longer than its overlapping entity
        i = 0
        for entity in entities_copy:
            j = 0
            for overlapping_entity in entities_copy:
                # Skip self
                if i != j:
                    e_start, e_end, oe_start, oe_end = entity[0], entity[
                        1], overlapping_entity[0], overlapping_entity[1]
                    # Delete any entity that overlaps, keep if longer
                    if ((e_start >= oe_start and e_start <= oe_end)
                            or (e_end <= oe_end and e_end >= oe_start)) \
                            and ((e_end - e_start) <= (oe_end - oe_start)):
                        entities.remove(entity)
                j += 1
            i += 1
        clean_data.append((text, {'entities': entities}))

    return clean_data


# train_spacy()

print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)


docx1 = nlp2(u"Govardhana K\nSenior Software Engineer\n\nBengaluru, Karnataka, Karnataka - Email me on Indeed: indeed.com/r/Govardhana-K/\nb2de315d95905b68\n\nTotal IT experience 5 Years 6 Months")
for token in docx1.ents:
    print(token.text, token.start_char, token.end_char, token.label_)
