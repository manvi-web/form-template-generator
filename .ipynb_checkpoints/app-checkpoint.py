from flask import Flask, request, jsonify
import sqlite3
import pickle
import json
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
mlb = pickle.load(open('mlb.pkl', 'rb'))  # <-- important!
app = Flask(__name__)
def init_db():
    conn = sqlite3.connect('templates.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            form_name TEXT,
            template_json TEXT
        )
    ''')
    conn.commit()
    conn.close()
@app.route('/generate_template', methods=['POST'])
def generate_template():
    data = request.get_json()
    form_name = data.get('form_name')

    if not form_name:
        return jsonify({"error": "form_name is required"}), 400

    
    vect_form = vectorizer.transform([form_name])
    y_pred = model.predict(vect_form)
    fields = mlb.inverse_transform(y_pred)[0]  # <-- convert binary prediction to field names

    
    template = {
        "form_name": form_name,
        "fields": list(fields)
    }
    template_json = json.dumps(template)

    
    conn = sqlite3.connect('templates.db')
    c = conn.cursor()
    c.execute('INSERT INTO templates (form_name, template_json) VALUES (?, ?)', (form_name, template_json))
    conn.commit()
    conn.close()

    return jsonify({"message": "Template generated successfully!", "template": template})

if __name__ == "__main__":
    init_db()  
    app.run(debug=True, port=10000)
