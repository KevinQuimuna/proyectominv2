import os
import cv2 as cv
import numpy as np
import base64
from flask import Flask, render_template, request, url_for
from PIL import Image
from io import BytesIO
import logging
import time

# Configurar logging para producción
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configurar carpeta de uploads para Render
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Archivos de modelos
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

# Cargar redes con manejo de errores mejorado
try:
    ageNet = cv.dnn.readNet(ageModel, ageProto)
    genderNet = cv.dnn.readNet(genderModel, genderProto)
    faceNet = cv.dnn.readNet(faceModel, faceProto)
    
    # Configurar para usar CPU (más estable en Render)
    ageNet.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    ageNet.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    genderNet.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    genderNet.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    faceNet.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
    faceNet.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    
    logger.info("Modelos cargados exitosamente en CPU.")
except Exception as e:
    logger.error(f"Error al cargar los modelos: {e}")
    raise

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Masculino', 'Femenino']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(image_path):
    try:
        frame = cv.imread(image_path)
        if frame is None:
            logger.error(f"No se pudo leer la imagen desde {image_path}")
            return "Error: no se pudo leer la imagen."

        logger.info(f"Procesando imagen con shape: {frame.shape}")
        
        # Crear blob para detección de rostros
        blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False)
        faceNet.setInput(blob)
        detections = faceNet.forward()
        
        faces_found = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                faces_found += 1
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Asegurar que las coordenadas estén dentro de los límites
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                # Crear blob para predicción
                blob_face = cv.dnn.blobFromImage(face, 1.0, (227, 227), [104, 117, 123], swapRB=False)

                # Predicción de género
                genderNet.setInput(blob_face)
                gender_preds = genderNet.forward()
                gender = GENDER_LIST[gender_preds[0].argmax()]
                gender_conf = gender_preds[0].max()

                # Predicción de edad
                ageNet.setInput(blob_face)
                age_preds = ageNet.forward()
                age = AGE_LIST[age_preds[0].argmax()]
                age_conf = age_preds[0].max()

                logger.info(f"Predicción: {gender} (conf: {gender_conf:.3f}), {age} (conf: {age_conf:.3f})")
                return f"{gender}, {age}"

        if faces_found == 0:
            return "No se detectó ningún rostro en la imagen."
        else:
            return "No se pudo procesar el rostro detectado."
            
    except Exception as e:
        logger.error(f"Error en la predicción: {e}")
        return f"Error en el procesamiento: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result = None
    filename = None

    if request.method == "POST":
        logger.info("Solicitud POST recibida")
        
        if 'image' not in request.files:
            error = "No se seleccionó ningún archivo."
            return render_template("index.html", error=error, result=result, filename=filename)

        file = request.files['image']
        if file.filename == '':
            error = "No se seleccionó ningún archivo."
            return render_template("index.html", error=error, result=result, filename=filename)

        if file and allowed_file(file.filename):
            # Generar nombre único para evitar colisiones
            timestamp = int(time.time())
            file_ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"upload_{timestamp}.{file_ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            logger.info(f"Imagen guardada en: {filepath}")
            result = predict(filepath)
        else:
            error = "Formato no soportado. Usa png, jpg, jpeg o bmp."

    return render_template("index.html", error=error, result=result, filename=filename)

@app.route("/camera", methods=["POST"])
def camera():
    logger.info("Solicitud POST recibida en /camera")
    image_data = request.form.get('imageData')
    
    if not image_data:
        error = "No se recibió imagen de la cámara."
        return render_template("index.html", error=error, result=None, filename=None)

    try:
        # Procesar imagen base64
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Guardar con nombre único
        timestamp = int(time.time())
        filename = f"camera_capture_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)
        
        logger.info(f"Imagen de cámara guardada en: {filepath}")
        result = predict(filepath)
        return render_template("index.html", result=result, filename=filename, error=None)
        
    except Exception as e:
        logger.error(f"Error al procesar imagen de cámara: {e}")
        error = "Error al procesar imagen desde la cámara."
        return render_template("index.html", error=error, result=None, filename=None)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # En producción, usar debug=False
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)