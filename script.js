// Загрузка модели
let model;
async function loadModel() {
    model = await tf.loadLayersModel('model/model.json');
    console.log('Model loaded');
}

// Функция для обработки изображения
function preprocessImage(img) {
    return tf.browser.fromPixels(img)
        .resizeNearestNeighbor([128, 128])
        .toFloat()
        .div(tf.scalar(255))
        .expandDims(0);
}

// Функция для классификации изображения
async function classifyImage(img) {
    const tensor = preprocessImage(img);
    const prediction = model.predict(tensor);
    const predictedClass = prediction.argMax(-1).dataSync()[0];
    const classNames = ["class_1", "class_2", "class_3", ..., "class_102"]; // Замените на ваши классы
    return classNames[predictedClass];
}

// Обработчик загрузки изображения
document.getElementById('uploadImage').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    const imgElement = document.getElementById('image');
    imgElement.src = URL.createObjectURL(file);

    imgElement.onload = async () => {
        const prediction = await classifyImage(imgElement);
        document.getElementById('prediction').innerText = prediction;
    };
});

// Загрузка модели при запуске
loadModel();
