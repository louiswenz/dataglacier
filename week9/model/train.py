import spacy
import json
import random
import logging
from sklearn.metrics import accuracy_score
import string
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from spacy.tokens import Doc
from spacy.training import Example
import matplotlib.pyplot as plt
import sys
import fitz
from nltk.corpus import stopwords
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import pos_tag
import numpy as np

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
            text = data['content']
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
                    # label = label.replace('-', ' ')
                    start, end = remove_whitespace_punctuation(
                        text, point['start'], point['end'] + 1)
                    entities.append((start, end, label))

                    # entities.append((point["start"], point["end"] + 1, label))

            training_data.append((text, {"entities": entities}))

        return training_data
    except Exception as e:
        logging.exception("Unable to process " +
                          dataturks_JSON_FilePath + "\n" + "error = " + str(e))
        return None


################### Train Spacy NER.###########


def train_spacy():
    loss = []
    score = []
    TRAIN_DATA = convert_dataturks_to_spacy(
        "dataset/EResume.json")

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
        for itn in range(20):
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                doc = nlp.make_doc(text)
                examples = [Example.from_dict(doc, annotations)]
                nlp.update(
                    examples,
                    drop=0.2,
                    sgd=optimizer,
                    losses=losses)
            sc = abs(1-(losses['ner'] / len(TRAIN_DATA)))
            print(losses)
            print("Score:", sc)
            loss.append(losses['ner'])
            score.append(sc)

    nlp.to_disk(Path(output_dir))
    print("Saved model to", output_dir)

    evaluate(nlp)

    return loss, score


def evaluate(nlp):
    examples = convert_dataturks_to_spacy("dataset/testdata.json")
    examples = clean_entities(examples)
    c = 0
    for text, annot in examples:

        f = open("resume"+str(c)+".txt", "w")
        doc_to_test = nlp(text)
        d = {}
        for ent in doc_to_test.ents:
            d[ent.label_] = []
        for ent in doc_to_test.ents:
            d[ent.label_].append(ent.text)

        for i in set(d.keys()):

            f.write("\n\n")
            f.write(i + ":"+"\n")
            for j in set(d[i]):
                f.write(j.replace('\n', '')+"\n")
    print("Files written")
    # d = {}
    # for ent in doc_to_test.ents:
    #     d[ent.label_] = [0, 0, 0, 0, 0, 0]
    #     for ent in doc_to_test.ents:
    #         doc_gold_text = Doc(nlp.vocab, words=text.split())
    #         example = Example.from_dict(doc_gold_text, annot)
    #         gold = example.get_aligned_ner()
    #         for i, v in enumerate(gold):
    #             if v == None:
    #                 gold[i] = 'O'
    #         y_true = [ent.label_ if ent.label_ in x else 'Not ' +
    #                   ent.label_ for x in gold]
    #         y_pred = [x.ent_type_ if x.ent_type_ ==
    #                   ent.label_ else 'Not '+ent.label_ for x in doc_to_test]
    #         if (d[ent.label_][0] == 0) and (len(y_true) == len(y_pred)):
    #             # f.write("For Entity "+ent.label_+"\n")
    #             # f.write(classification_report(y_true, y_pred)+"\n")
    #             (p, r, f, s) = precision_recall_fscore_support(
    #                 y_true, y_pred, average='weighted')
    #             a = accuracy_score(y_true, y_pred)
    #             d[ent.label_][0] = 1
    #             d[ent.label_][1] += p
    #             d[ent.label_][2] += r
    #             d[ent.label_][3] += f
    #             d[ent.label_][4] += a
    #             d[ent.label_][5] += 1
    #     c += 1
    # for i in d:
    #     print("\n For Entity "+i+"\n")
    #     print("Accuracy : "+str((d[i][4]/d[i][5])*100)+"%")
    #     print("Precision : "+str(d[i][1]/d[i][5]))
    #     print("Recall : "+str(d[i][2]/d[i][5]))
    #     print("F-score : "+str(d[i][3]/d[i][5]))
    return


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
        # text = text.replace('-', ' ')
        # text = ' '.join(text.split('\n'))
        clean_data.append((text, {'entities': entities}))

    return clean_data


# loss, score = train_spacy()

# plt.plot(loss, label='loss')
# plt.legend()
# plt.show()
# plt.plot(score, label='score')
# plt.legend()
# plt.show()

# print("Loading from", output_dir)
# nlp2 = spacy.load(output_dir)
# nlp = spacy.load('en_core_web_sm')

# docx1 = nlp2(u"Govardhana K\nSenior Software Engineer\n\nBengaluru, Karnataka, Karnataka - Email me on Indeed: indeed.com/r/Govardhana-K/\nb2de315d95905b68\n\nTotal IT experience 5 Years 6 Months")
# for token in docx1.ents:
#     print(token.text, token.start_char, token.end_char, token.label_)


# TRAIN_DATA = convert_dataturks_to_spacy(
#     "dataset/Resumeold.json")

# TRAIN_DATA = clean_entities(TRAIN_DATA)
# print(TRAIN_DATA[0])

# pdf
# filename = "dataset/ZongdaoWen2023.pdf"
# doc = fitz.open(filename)
# text = ''
# for page in doc:
#     text = text + str(page.get_text())
# # text = text.strip()
# # text = ' '.join(text.split('\n'))

# docx1 = nlp2(text)
# for token in docx1.ents:
#     print(token.text, token.start_char, token.end_char, token.label_)


# EDA
def plot_top_keywords(text):
    # Tokenize the text into individual words
    words = nltk.word_tokenize(text.lower())

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha()
                      and word not in stop_words]

    # Calculate word frequency
    freq_dist = nltk.FreqDist(filtered_words)

    # Get the top 10 most common keywords
    top_keywords = freq_dist.most_common(10)

    # Extract the keywords and their frequencies
    keywords = [keyword[0] for keyword in top_keywords]
    frequencies = [keyword[1] for keyword in top_keywords]

    # Plot the keyword frequencies
    plt.bar(range(len(keywords)), frequencies)
    plt.xticks(range(len(keywords)), keywords)
    plt.xlabel('Keywords')
    plt.ylabel('Frequency')
    plt.title('Top 10 Most Used Keywords')
    plt.show()


def analyze_sentence_lengths(text):
    text = text.split("@sp")
    # Tokenize the text into sentences
    # sentences = nltk.sent_tokenize(text)
    # Get the length of each sentence
    sentence_lengths = [len(res)
                        for res in text]
    avg = np.mean(sentence_lengths)

    # Plot a histogram of sentence lengths
    plt.hist(sentence_lengths)
    plt.xlabel('Resume Length(Char)')
    plt.ylabel('Count')
    plt.title('Resume Length Distribution')
    plt.show()


def perform_sentiment_analysis(text):
    # Initialize the sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Analyze sentiment for each sentence
    sentiment_scores = []
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        sentiment_scores.append(sid.polarity_scores(sentence)['compound'])

    # Plot the sentiment scores
    plt.plot(sentiment_scores)
    plt.xlabel('Sentence Index')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Analysis')
    plt.show()


def analyze_pos_tags(text):
    # Tokenize the text into individual words
    words = nltk.word_tokenize(text)

    # Perform POS tagging
    pos_tags = nltk.pos_tag(words)

    # Count the frequency of each POS tag
    tag_freq = nltk.FreqDist(tag for word, tag in pos_tags)

    # Extract the tags and their frequencies
    tags = tag_freq.keys()
    frequencies = tag_freq.values()

    # Plot the tag frequencies
    plt.bar(tags, frequencies)
    plt.xlabel('POS Tags')
    plt.ylabel('Frequency')
    plt.title('POS Tagging Distribution')
    plt.xticks(rotation=45)
    plt.show()


# nltk.download('punkt')
# nltk.download('vader_lexicon')
# nltk.download('averaged_perceptron_tagger')

text_all = ''
labels = []
with open('dataset/Resumeold.json', 'r') as f:
    lines = f.readlines()

for line in lines:
    data = json.loads(line)
    text = data['content']
    annotation = data['annotation']
    for x in annotation:
        labels.extend(x['label'])
    text_all += text + '@sp'
text_all = text_all.replace('\n', '')
labels = list(set(labels))

# plot_top_keywords(text_all)
analyze_sentence_lengths(text_all)
# perform_sentiment_analysis(text_all)
# analyze_pos_tags(text_all)

print(labels)
