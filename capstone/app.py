from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import pandas as pd
import gradio as gr
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

CORS(app, resources={
    r"/recommend": {
        "origins": ["http://localhost:5173", "http://localhost:5000"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load model
model = pickle.load(open("model.pkl", "rb"))

def predict(text):
    return model.predict([text])[0]

gr.Interface(fn=predict, inputs="text", outputs="text").launch()

# Load model dan tokenizer dengan pengecekan error
try:
    model = tf.keras.models.load_model('multilabel_model.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer = None

# Load data wisata dengan pengecekan file
try:
    df = pd.read_excel('java.xlsx')
    print("java.xlsx loaded successfully")
    required_columns = ['place_name', 'province', 'deskripsi', 'gambar']
    if not all(col in df.columns for col in required_columns):
        print(f"Missing columns in java.xlsx. Required: {required_columns}")
        df = pd.DataFrame(columns=required_columns)
except FileNotFoundError:
    print("Error: java.xlsx not found")
    df = pd.DataFrame(columns=['place_name', 'province', 'deskripsi', 'gambar'])
except Exception as e:
    print(f"Error loading java.xlsx: {e}")
    df = pd.DataFrame(columns=['place_name', 'province', 'deskripsi', 'gambar'])

max_len = 50
all_labels = ['alam', 'buatan', 'budaya', 'religi', 'edukasi']

def predict_labels(text):
    if not model or not tokenizer:
        return {label: 0 for label in all_labels}, {label: 0.0 for label in all_labels}
    try:
        seq = tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)
        preds = model.predict(padded)[0]
        labels_bin = {label: int(preds[i] >= 0.5) for i, label in enumerate(all_labels)}
        labels_prob = {label: float(preds[i]) for i, label in enumerate(all_labels)}
        return labels_bin, labels_prob
    except Exception as e:
        print(f"Error predicting labels: {e}")
        return {label: 0 for label in all_labels}, {label: 0.0 for label in all_labels}

def recommend_places(user_province, user_labels, user_description, df, top_n=50, threshold=0.10):
    if df.empty:
        return {"error": "No data available. Check java.xlsx."}
    
    filtered_df = df[df['province'].str.lower() == user_province.lower()]
    if filtered_df.empty:
        return {"error": "Maaf, tidak ada tempat di provinsi tersebut."}

    # Jika selected_labels kosong, gunakan prediksi label dari model
    selected_labels = [label for label, val in user_labels.items() if val == 1]
    if not selected_labels:
        _, predicted_labels = predict_labels(user_description)
        selected_labels = [label for label, val in predicted_labels.items() if val >= 0.5]
        if not selected_labels:
            return {"error": "Tidak ada label yang diprediksi dari deskripsi. Mohon pilih minimal satu tipe wisata."}

    mask = filtered_df[selected_labels].sum(axis=1) > 0
    filtered_df = filtered_df[mask]
    if filtered_df.empty:
        return {"error": "Maaf, tidak ada tempat dengan tipe wisata tersebut di provinsi itu."}

    combined_text = (filtered_df['place_name'] + ' ' + filtered_df['deskripsi']).tolist()
    corpus_with_query = combined_text + [user_description]

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus_with_query)
        cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        top_indices = cosine_sim.argsort()[::-1][:top_n]
        top_indices = [idx for idx in top_indices if cosine_sim[idx] >= threshold]

        if len(top_indices) == 0:
            return {"error": "Maaf, tidak ada tempat dengan kemiripan yang cukup di provinsi tersebut."}

        recommendations = []
        for idx in top_indices:
            place = filtered_df.iloc[idx]
            print(f"Processing place: {place['place_name']}, gambar: {place['gambar']}")
            image_url = f"http://localhost:5000/gambar/{place['gambar']}" if place['gambar'] else "https://source.unsplash.com/300x200/?travel"
            recommendations.append({
                'id': int(place['id']),
                'place_name': place['place_name'],
                'province': place['province'],
                'deskripsi': place['deskripsi'],
                'gambar': image_url,
                'similarity_score': float(cosine_sim[idx])
            })
        return recommendations
    except Exception as e:
        print(f"Error in recommend_places: {e}")
        return {"error": f"Internal error: {str(e)}"}

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_province = data.get('province', '').strip()
        user_description = data.get('description', '').strip()
        user_selected_labels = data.get('selected_labels', [])  # list of label strings
        threshold = data.get('threshold', 0.10)

        print(f"Received data: {data}")

        if not user_province or not user_description:
            return jsonify({"error": "Province and description must be provided."}), 400

        selected_labels = {label: 1 if label in user_selected_labels else 0 for label in all_labels}
        predicted_labels_bin, predicted_labels_prob = predict_labels(user_description)
        result = recommend_places(user_province, selected_labels, user_description, df, threshold=threshold)

        print("Recommendations response:", result)

        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 400  # Ubah ke 400 untuk validasi error

        return jsonify({
            "input_labels": selected_labels,
            "predicted_labels_prob": predicted_labels_prob,
            "recommendations": result
        })
    except Exception as e:
        print(f"Error in /recommend endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5001)