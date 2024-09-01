import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs/promises';
import path from 'path';

export default async function handler(req, res) {
    try {
        // Cargar el modelo desde la carpeta public
        const modelPath = path.join(process.cwd(), 'public', 'model.json');
        const model = await tf.loadGraphModel(`file://${modelPath}`);

        // Cargar el scaler y el label encoder
        const scalerPath = path.join(process.cwd(), 'public', 'scaler.json');
        const labelEncoderPath = path.join(process.cwd(), 'public', 'label_encoder.json');

        const [scaler, labelEncoder] = await Promise.all([
            fs.readFile(scalerPath, 'utf8').then(JSON.parse),
            fs.readFile(labelEncoderPath, 'utf8').then(JSON.parse)
        ]);

        // Obtener parámetros de consulta
        const { hum, luz, pres, temp, vel } = req.query;

        // Validación de parámetros
        if ([hum, luz, pres, temp, vel].some(val => isNaN(parseFloat(val)))) {
            return res.status(400).json({ error: 'Invalid input parameters. Ensure all parameters are numbers.' });
        }

        // Preparar la entrada para la predicción
        const input = [parseFloat(hum), parseFloat(luz), parseFloat(pres), parseFloat(temp), parseFloat(vel)];
        const scaledInput = input.map((val, i) => (val - scaler.mean[i]) / scaler.scale[i]);

        // Realizar la predicción
        const tensorInput = tf.tensor2d([scaledInput], [1, 5]);
        const prediction = model.predict(tensorInput);
        const predClass = prediction.argMax(-1).dataSync()[0];
        const result = labelEncoder.classes[predClass];

        // Devolver la predicción
        return res.status(200).json({ prediction: result });

    } catch (error) {
        console.error('Error during prediction:', error);
        return res.status(500).json({ error: `Error during prediction: ${error.message}` });
    }
}
