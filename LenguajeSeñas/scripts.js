document.addEventListener('DOMContentLoaded', async function () {
  const webcamElement = document.getElementById('webcam');
  const startButton = document.getElementById('start-button');
  const stopButton = document.getElementById('stop-button');
  const lastGestureElement = document.getElementById('last-gesture');
  const consoleElement = document.getElementById('console');
  const letterButtons = document.querySelectorAll('#letter-buttons button');
  const saveButton = document.getElementById('save-button');
  const loadButton = document.getElementById('load-button');
  const loadButtonTrigger = document.getElementById('load-button-trigger');

  const labelToLetter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'];

  let net;
  let knn;
  let webcam;
  let recognitionActive = true;

  // Función para cargar el modelo
  async function loadModel() {
    try {
      net = await mobilenet.load();
      knn = knnClassifier.create();
    } catch (error) {
      console.error('Error al cargar el modelo:', error);
    }
  }

  async function addExample(label) {
    try {
      const activation = net.infer(webcamElement, 'conv_preds');
      console.log('Activación capturada:', activation);
      knn.addExample(activation, label);
      console.log(`Ejemplo agregado para la etiqueta: ${label}`);
    } catch (error) {
      console.error('Error al añadir el ejemplo:', error);
    }
  }

  letterButtons.forEach(button => {
    button.addEventListener('click', () => {
      const label = button.getAttribute('data-letter');
      addExample(label);
    });
  });

  async function setupWebcam() {
    return new Promise((resolve, reject) => {
      const navigatorAny = navigator;
      navigator.getUserMedia =
        navigator.getUserMedia ||
        navigatorAny.webkitGetUserMedia ||
        navigatorAny.mozGetUserMedia ||
        navigatorAny.msGetUserMedia;

      if (navigator.getUserMedia) {
        navigator.getUserMedia(
          { video: true },
          (stream) => {
            webcamElement.srcObject = stream;
            webcam = stream;
            resolve();
          },
          (error) => {
            reject();
          }
        );
      } else {
        reject();
      }
    });
  }

  async function init() {
    console.log('Iniciando la aplicación...');
    await setupWebcam();
    console.log('Cámara iniciada.');
    await loadModel();
    console.log('Modelo cargado.');
    startButton.removeAttribute('disabled');
  }

  init();


  startButton.addEventListener('click', () => {
    detectGesture();
  });

  // Event listener para el botón de detener reconocimiento
  stopButton.addEventListener('click', () => {
    console.log('Deteniendo el reconocimiento manualmente.');
    recognitionActive = false;
  });

  saveButton.addEventListener('click', async () => {
    try {
      const dataset = knn.getClassifierDataset();
      const tensorObj = { data: dataset };
      const tensorJSON = JSON.stringify(tensorObj);
      const blob = new Blob([tensorJSON], { type: 'application/json' });
      const href = await URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = href;
      link.download = 'knn-model.json';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Error al guardar el modelo:', error);
    }
  });

  loadButtonTrigger.addEventListener('click', () => {
    loadButton.click();
  });

  loadButton.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = async (event) => {
        try {
          const modelJson = JSON.parse(event.target.result);
          console.log('Contenido del modelo:', modelJson);
  
          // Asegúrate de que la estructura del modelo sea la correcta
          if (modelJson && modelJson.data) {
            knn.setClassifierDataset(modelJson.data);
          } else {
            console.error('Error: Datos del modelo no definidos o no presentes.');
          }
        } catch (error) {
          console.error('Error al cargar el modelo:', error);
        }
      };
      reader.readAsText(file);
    }
  });
  

  async function detectGesture() {
    if (!webcamElement.srcObject) {
      console.error('La cámara no está lista.');
      return;
    }

    if (knn.getNumClasses() > 0 && recognitionActive) {
      try {
        while (recognitionActive) {
          const activation = net.infer(webcamElement, 'conv_preds');
          const result = await knn.predictClass(activation);

          const label = result.label;

          console.log('Etiqueta predicha:', label);

          if (label >= 0 && label < labelToLetter.length) {
            const gesture = labelToLetter[label];
            console.log('Letra correspondiente:', {gesture});

            if (lastGestureElement && consoleElement) {
              lastGestureElement.innerText = gesture;
              consoleElement.innerText = `Último gesto detectado: ${gesture}`;
            } else {
              console.error('Los elementos del DOM no están presentes.');
              recognitionActive = false;
              return;
            }
          } else {
            console.error('Etiqueta predicha fuera del rango.');
          }

          if (result.iterations >= 10) {
            console.log('Deteniendo el reconocimiento después de 10 iteraciones.');
            recognitionActive = false;
            return;
          }
        }
      } catch (error) {
        console.error('Error al predecir:', error);
        recognitionActive = false;
      }
    } else {
      console.error('No hay ejemplos en el clasificador o el reconocimiento ya se detuvo. Agrega ejemplos antes de predecir o reinicia el reconocimiento.');
    }
  }
});
