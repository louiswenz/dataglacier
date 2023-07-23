import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import textract
import spacy

app = Flask(__name__, static_url_path='/static')
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
nlp = spacy.load('en_core_web_sm')

output_dir = "/Users/Louis/Desktop/DataGlacier/dataglacier/week11/nlp_model"
# nlp2 = spacy.load(output_dir)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_information(file_path):
    text = textract.process(file_path).decode('utf-8')
    text = text.strip()
    text = ' '.join(text.split('\n'))
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file selected')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            entities = extract_information(file_path)
            os.remove(file_path)  # Delete the uploaded file

            return render_template('index.html', entities=entities)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
