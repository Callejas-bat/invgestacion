import * as tf from '@tensorflow/tfjs';
import path from 'path';
import fs from 'fs';

function loadJSONSync(filePath) {
    const fileContent = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(fileContent);
}

function applyScaler(input, scaler) {
    return input.map((val, i) => (val - scaler.mean[i]) / scaler.scale[i]);
}

function inverseLabelEncode(pred, labelEncoder) {
    return labelEncoder.classes[pred];
}

export default async function handler(req, res) {
    try {
        // Cargar los archivos JSON desde la carpeta public
        const modelPath = path.join(process.cwd(), 'public', 'model.json');
        const scalerPath = path.join(process.cwd(), 'public', 'scaler.json');
        const labelEncoderPath = path.join(process.cwd(), 'public', 'label_encoder.json');

        // Cargar el modelo de TensorFlow
        const model = await tf.loadGraphModel(`file://${modelPath}`);
        
        // Cargar el scaler y el label encoder
        const scaler = loadJSONSync(scalerPath);
        const labelEncoder = loadJSONSync(labelEncoderPath);

        // Obtener los parámetros de la consulta
        const { hum, luz, pres, temp, vel } = req.query;

        // Validar que todos los parámetros están presentes y son números
        if ([hum, luz, pres, temp, vel].some(val => isNaN(parseFloat(val)))) {
            return res.status(400).json({ error: 'Invalid input parameters. Ensure all parameters are numbers.' });
        }

        // Procesar la entrada y hacer la predicción
        const input = [parseFloat(hum), parseFloat(luz), parseFloat(pres), parseFloat(temp), parseFloat(vel)];
        const scaledInput = applyScaler(input, scaler);

        const tensorInput = tf.tensor2d([scaledInput], [1, 5]);
        const prediction = model.predict(tensorInput);
        const predClass = prediction.argMax(-1).dataSync()[0];
        const result = inverseLabelEncode(predClass, labelEncoder);

        // Devolver la respuesta en formato JSON
        return res.status(200).json({ prediction: result });

    } catch (error) {
        console.error('Error during prediction:', error);
        return res.status(500).json({ error: 'Error during prediction.' });
    }
}
