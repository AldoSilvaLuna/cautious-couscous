from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os

# Configuración básica
dir_base = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(dir_base, 'uploaded_images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Crear carpeta si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/upload/batch', methods=['POST'])
def upload_batch():
    print("upload_batch")
    # Obtiene la lista de archivos con el campo 'images'
    files = request.files.getlist('images')
    if not files:
        response = jsonify(success=False, message='No se encontraron archivos con key "images"'), 400
        print('upload_batch response', response)
        return response

    saved_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            saved_files.append(filename)
        else:
            # Si algún archivo no cumple el formato permitido, omitirlo
            continue

    return jsonify(success=True,
                   message=f'Se guardaron {len(saved_files)} archivos',
                   files=saved_files)


if __name__ == '__main__':
    # Ejecutar con: python3 upload_service.py
    app.run(host='0.0.0.0', port=5000, debug=True)