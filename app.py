from flask import Flask,render_template, request
import pickle


app = Flask(__name__,template_folder='M:/PLAGproject/templates')

model = pickle.load(open('M:/PLAGproject/.venv/model.pk1','rb'))
tfidf_vectorizer = pickle.load(open('M:/PLAGproject/.venv/tfidf_vectorizer.pk1','rb'))

def detect(input_text):
    vectorized_text = tfidf_vectorizer.transform([input_text])
    result = model.predict(vectorized_text)
    return "Plagiarism Detected" if result[0] == 1 else "No Plagiarism Detected"


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    input_text = request.form['text']
    detection_result = detect(input_text)
    return render_template('index.html', result=detection_result)



if __name__=='__main__':
    app.run(debug=True)