export default async function handler(req, res) {
    try {
        // Cargar TensorFlow.js desde un CDN
        await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs');

        // Cargar los modelos y otros archivos JSON desde URLs
        const modelUrl = 'https://invgestacion.vercel.app/public/model.json';
        const model = await tf.loadGraphModel(modelUrl);
        
        const scalerUrl = 'https://invgestacion.vercel.app/public/scaler.json';
        const scalerResponse = await fetch(scalerUrl);
        const scaler = await scalerResponse.json();

        const labelEncoderUrl = 'https://invgestacion.vercel.app/public/label_encoder.json';
        const labelEncoderResponse = await fetch(labelEncoderUrl);
        const labelEncoder = await labelEncoderResponse.json();

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
        // Capturar y devolver errores detallados
        console.error('Error during prediction:', error);
        return res.status(500).json({ error: `Error during prediction: ${error.message}` });
    }
}
