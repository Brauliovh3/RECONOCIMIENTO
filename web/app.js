/* ============================================================
   HELMET AI | SENATI
   Clasificación YOLO + ONNX Runtime Web
============================================================ */

"use strict";


/* ============================================================
   CONFIGURACIÓN
============================================================ */

const MODEL_PATH = "./model/best.onnx";

const INPUT_SIZE = 224;

const INFERENCE_INTERVAL = 220;

const SMOOTHING_FRAMES = 5;

const HELMET_MIN_CONFIDENCE = 0.60;

const VERIFY_CONFIDENCE = 0.50;


/*
    El modelo V4 fue entrenado como:

    0 = casco
    1 = sin_casco

    Ultralytics normalmente conserva los nombres
    en metadata, pero usamos también este orden
    como respaldo.
*/

const CLASS_NAMES = [
    "casco",
    "sin_casco"
];


/* ============================================================
   ELEMENTOS DOM
============================================================ */

const video =
    document.getElementById("video");

const overlay =
    document.getElementById("overlay");

const cameraContainer =
    document.getElementById("cameraContainer");

const cameraPlaceholder =
    document.getElementById("cameraPlaceholder");

const cameraStatus =
    document.getElementById("cameraStatus");

const fpsCounter =
    document.getElementById("fpsCounter");

const cameraAdvice =
    document.getElementById("cameraAdvice");

const startButton =
    document.getElementById("startButton");

const stopButton =
    document.getElementById("stopButton");

const resultCard =
    document.getElementById("resultCard");

const resultIcon =
    document.getElementById("resultIcon");

const resultState =
    document.getElementById("resultState");

const resultMessage =
    document.getElementById("resultMessage");

const confidenceValue =
    document.getElementById("confidenceValue");

const confidenceFill =
    document.getElementById("confidenceFill");

const adviceText =
    document.getElementById("adviceText");

const hologram =
    document.getElementById("hologram");

const targetIcon =
    document.getElementById("targetIcon");

const trackingLabel =
    document.getElementById("trackingLabel");

const liveIndicator =
    document.getElementById("liveIndicator");


/* ============================================================
   ESTADO
============================================================ */

let session = null;

let stream = null;

let running = false;

let inferenceTimer = null;

let lastInferenceTime = 0;

let fpsFrames = 0;

let fpsLastTime = performance.now();

let currentFPS = 0;

let predictionHistory = [];

let lastState = "neutral";

let lastSoundTime = 0;


/* ============================================================
   CANVAS
============================================================ */

const ctx =
    overlay.getContext("2d");


/* ============================================================
   UTILIDADES
============================================================ */

function clamp(value, min, max) {

    return Math.max(
        min,
        Math.min(max, value)
    );
}


function percent(value) {

    return `${(
        clamp(value, 0, 1) * 100
    ).toFixed(1)}%`;
}


function sleep(ms) {

    return new Promise(
        resolve => setTimeout(resolve, ms)
    );
}


/* ============================================================
   RESIZE CANVAS
============================================================ */

function resizeCanvas() {

    if (!video.videoWidth || !video.videoHeight) {
        return;
    }

    const rect =
        video.getBoundingClientRect();

    const dpr =
        window.devicePixelRatio || 1;

    overlay.width =
        Math.round(rect.width * dpr);

    overlay.height =
        Math.round(rect.height * dpr);

    overlay.style.width =
        `${rect.width}px`;

    overlay.style.height =
        `${rect.height}px`;

    ctx.setTransform(
        dpr,
        0,
        0,
        dpr,
        0,
        0
    );
}


window.addEventListener(
    "resize",
    resizeCanvas
);


/* ============================================================
   AUDIO
============================================================ */

let audioContext = null;


function getAudioContext() {

    if (!audioContext) {

        const AudioContext =
            window.AudioContext ||
            window.webkitAudioContext;

        if (!AudioContext) {
            return null;
        }

        audioContext =
            new AudioContext();
    }

    return audioContext;
}


function playTone(
    frequency,
    duration,
    delay = 0,
    volume = 0.045
) {

    const audio =
        getAudioContext();

    if (!audio) {
        return;
    }

    if (audio.state === "suspended") {
        audio.resume();
    }

    const oscillator =
        audio.createOscillator();

    const gain =
        audio.createGain();

    oscillator.type = "sine";

    oscillator.frequency.value =
        frequency;

    gain.gain.setValueAtTime(
        0,
        audio.currentTime + delay
    );

    gain.gain.linearRampToValueAtTime(
        volume,
        audio.currentTime + delay + 0.015
    );

    gain.gain.exponentialRampToValueAtTime(
        0.001,
        audio.currentTime +
        delay +
        duration
    );

    oscillator.connect(gain);
    gain.connect(audio.destination);

    oscillator.start(
        audio.currentTime + delay
    );

    oscillator.stop(
        audio.currentTime +
        delay +
        duration +
        0.02
    );
}


function playHelmetSound() {

    playTone(
        740,
        0.12,
        0,
        0.04
    );

    playTone(
        980,
        0.14,
        0.14,
        0.035
    );
}


function playAlertSound() {

    playTone(
        280,
        0.16,
        0,
        0.05
    );

    playTone(
        210,
        0.20,
        0.17,
        0.045
    );
}


function playSoundForState(state) {

    const now =
        Date.now();

    if (
        now - lastSoundTime <
        2500
    ) {
        return;
    }

    if (
        state !== "helmet" &&
        state !== "nohelmet"
    ) {
        return;
    }

    lastSoundTime = now;

    if (state === "helmet") {
        playHelmetSound();
    } else {
        playAlertSound();
    }
}


/* ============================================================
   CAMERA
============================================================ */

async function startCamera() {

    if (running) {
        return;
    }

    try {

        cameraAdvice.textContent =
            "Solicitando acceso a la cámara...";

        cameraStatus.textContent =
            "CONECTANDO";

        stream =
            await navigator.mediaDevices.getUserMedia({

                video: {
                    facingMode: "user",

                    width: {
                        ideal: 1280
                    },

                    height: {
                        ideal: 720
                    }
                },

                audio: false
            });


        video.srcObject =
            stream;


        await video.play();


        await waitForVideoMetadata();


        resizeCanvas();


        running = true;


        cameraPlaceholder.style.display =
            "none";

        startButton.disabled =
            true;

        stopButton.disabled =
            false;


        cameraStatus.textContent =
            "CÁMARA ACTIVA";

        liveIndicator.classList.add(
            "active"
        );

        liveIndicator.innerHTML =
            "<span></span> ONLINE";


        cameraAdvice.textContent =
            "Mantén el casco visible frente a la cámara.";


        resetPrediction();


        inferenceTimer =
            setInterval(
                runInference,
                INFERENCE_INTERVAL
            );


        runInference();


    } catch (error) {

        console.error(
            "Error de cámara:",
            error
        );

        stopCamera();


        cameraStatus.textContent =
            "ERROR DE CÁMARA";

        cameraAdvice.textContent =
            getCameraErrorMessage(error);


        resultState.textContent =
            "CÁMARA NO DISPONIBLE";

        resultMessage.textContent =
            getCameraErrorMessage(error);

    }
}


function waitForVideoMetadata() {

    return new Promise(
        resolve => {

            if (
                video.readyState >= 2 &&
                video.videoWidth > 0
            ) {

                resolve();

                return;
            }


            const handler = () => {

                video.removeEventListener(
                    "loadedmetadata",
                    handler
                );

                resolve();
            };


            video.addEventListener(
                "loadedmetadata",
                handler
            );

        }
    );
}


function getCameraErrorMessage(error) {

    if (!error) {

        return "No fue posible acceder a la cámara.";
    }


    if (
        error.name ===
        "NotAllowedError"
    ) {

        return (
            "Permiso de cámara rechazado. " +
            "Permite el acceso a la cámara desde el navegador."
        );
    }


    if (
        error.name ===
        "NotFoundError"
    ) {

        return (
            "No se encontró ninguna cámara disponible."
        );
    }


    if (
        error.name ===
        "NotReadableError"
    ) {

        return (
            "La cámara está siendo utilizada por otra aplicación."
        );
    }


    if (
        error.name ===
        "SecurityError"
    ) {

        return (
            "El navegador bloqueó el acceso por seguridad. " +
            "Usa HTTPS o localhost."
        );
    }


    return (
        "No fue posible iniciar la cámara. " +
        "Revisa los permisos del navegador."
    );
}


function stopCamera() {

    running = false;


    if (inferenceTimer) {

        clearInterval(
            inferenceTimer
        );

        inferenceTimer = null;
    }


    if (stream) {

        stream
            .getTracks()
            .forEach(
                track => track.stop()
            );

        stream = null;
    }


    video.srcObject = null;


    startButton.disabled =
        false;

    stopButton.disabled =
        true;


    cameraPlaceholder.style.display =
        "flex";


    cameraStatus.textContent =
        "CÁMARA INACTIVA";


    liveIndicator.classList.remove(
        "active"
    );


    liveIndicator.innerHTML =
        "<span></span> OFFLINE";


    cameraAdvice.textContent =
        "Inicia la cámara para comenzar el análisis.";


    clearOverlay();

    resetPrediction();


    resultCard.className =
        "result-card state-neutral";


    resultIcon.textContent =
        "AI";


    resultState.textContent =
        "SIN ANALIZAR";


    resultMessage.textContent =
        "Inicia la cámara para comenzar la detección.";


    confidenceValue.textContent =
        "0.0%";


    confidenceFill.style.width =
        "0%";


    adviceText.textContent =
        "Colócate frente a la cámara.";


    setHologramState(
        "neutral"
    );
}


function clearOverlay() {

    const rect =
        overlay.getBoundingClientRect();

    ctx.clearRect(
        0,
        0,
        rect.width,
        rect.height
    );
}


/* ============================================================
   MODELO
============================================================ */

async function loadModel() {

    cameraAdvice.textContent =
        "Cargando modelo de inteligencia artificial...";


    try {

        /*
            Configuración de ONNX Runtime.
            Para Vercel y navegador usamos WebAssembly.
        */

        ort.env.wasm.numThreads = 1;

        ort.env.wasm.simd = true;


        session =
            await ort.InferenceSession.create(
                MODEL_PATH,
                {
                    executionProviders: [
                        "wasm"
                    ]
                }
            );


        console.log(
            "Modelo ONNX cargado:",
            session
        );


        cameraAdvice.textContent =
            "Modelo IA listo. Puedes iniciar la cámara.";


    } catch (error) {

        console.error(
            "Error cargando ONNX:",
            error
        );


        cameraAdvice.textContent =
            "No se pudo cargar el modelo ONNX.";


        resultState.textContent =
            "ERROR DEL MODELO";


        resultMessage.textContent =
            "Verifica que model/best.onnx exista en el servidor.";


        throw error;
    }
}


/* ============================================================
   PREPROCESAMIENTO
============================================================ */

function preprocessFrame() {

    const canvas =
        document.createElement("canvas");

    canvas.width =
        INPUT_SIZE;

    canvas.height =
        INPUT_SIZE;


    const canvasContext =
        canvas.getContext(
            "2d",
            {
                willReadFrequently: true
            }
        );


    /*
        Clasificación V4:
        RGB
        224x224
        float32
        0-1
    */

    canvasContext.drawImage(
        video,
        0,
        0,
        INPUT_SIZE,
        INPUT_SIZE
    );


    const imageData =
        canvasContext.getImageData(
            0,
            0,
            INPUT_SIZE,
            INPUT_SIZE
        );


    const data =
        imageData.data;


    const area =
        INPUT_SIZE * INPUT_SIZE;


    const tensorData =
        new Float32Array(
            area * 3
        );


    for (
        let i = 0;
        i < area;
        i++
    ) {

        const source =
            i * 4;

        tensorData[i] =
            data[source] / 255;

        tensorData[area + i] =
            data[source + 1] / 255;

        tensorData[
            area * 2 + i
        ] =
            data[source + 2] / 255;
    }


    return new ort.Tensor(
        "float32",
        tensorData,
        [
            1,
            3,
            INPUT_SIZE,
            INPUT_SIZE
        ]
    );
}


/* ============================================================
   INFERENCIA
============================================================ */

async function runInference() {

    if (!running) {
        return;
    }

    if (!session) {
        return;
    }

    if (
        !video.videoWidth ||
        !video.videoHeight
    ) {
        return;
    }


    /*
        Evita ejecutar varias inferencias
        simultáneamente.
    */

    const now =
        performance.now();


    if (
        now - lastInferenceTime <
        INFERENCE_INTERVAL * 0.75
    ) {

        return;
    }


    lastInferenceTime =
        now;


    try {

        const input =
            preprocessFrame();


        const inputName =
            session.inputNames[0];


        const output =
            await session.run({
                [inputName]: input
            });


        const outputName =
            session.outputNames[0];


        const tensor =
            output[outputName];


        const scores =
            Array.from(
                tensor.data
            );


        const probabilities =
            softmax(scores);


        const prediction =
            getPrediction(
                probabilities
            );


        processPrediction(
            prediction
        );


        updateFPS();


    } catch (error) {

        console.error(
            "Error durante inferencia:",
            error
        );

        cameraAdvice.textContent =
            "Error procesando la imagen.";
    }
}


/* ============================================================
   SOFTMAX
============================================================ */

function softmax(values) {

    if (
        !Array.isArray(values) ||
        values.length === 0
    ) {

        return [];
    }


    const max =
        Math.max(...values);


    const exponentials =
        values.map(
            value =>
                Math.exp(
                    value - max
                )
        );


    const total =
        exponentials.reduce(
            (sum, value) =>
                sum + value,
            0
        );


    if (!total) {

        return values.map(
            () =>
                1 / values.length
        );
    }


    return exponentials.map(
        value =>
            value / total
    );
}


/* ============================================================
   PREDICCIÓN
============================================================ */

function getPrediction(
    probabilities
) {

    let bestIndex = 0;

    let bestProbability =
        probabilities[0] || 0;


    for (
        let i = 1;
        i < probabilities.length;
        i++
    ) {

        if (
            probabilities[i] >
            bestProbability
        ) {

            bestProbability =
                probabilities[i];

            bestIndex =
                i;
        }
    }


    return {

        index: bestIndex,

        label:
            CLASS_NAMES[bestIndex] ||
            `clase_${bestIndex}`,

        confidence:
            bestProbability

    };
}


/* ============================================================
   SMOOTHING
============================================================ */

function resetPrediction() {

    predictionHistory = [];

    lastState = "neutral";
}


function smoothPrediction(
    prediction
) {

    predictionHistory.push(
        prediction
    );


    if (
        predictionHistory.length >
        SMOOTHING_FRAMES
    ) {

        predictionHistory.shift();
    }


    const totals = {};


    for (
        const item of predictionHistory
    ) {

        if (
            !totals[item.index]
        ) {

            totals[item.index] = [];
        }


        totals[item.index].push(
            item.confidence
        );
    }


    let bestIndex =
        prediction.index;

    let bestAverage =
        0;


    for (
        const key in totals
    ) {

        const values =
            totals[key];


        const average =
            values.reduce(
                (sum, value) =>
                    sum + value,
                0
            ) /
            values.length;


        if (
            average >
            bestAverage
        ) {

            bestAverage =
                average;

            bestIndex =
                Number(key);
        }
    }


    return {

        index: bestIndex,

        label:
            CLASS_NAMES[bestIndex] ||
            `clase_${bestIndex}`,

        confidence:
            bestAverage

    };
}


/* ============================================================
   PROCESAMIENTO DEL RESULTADO
============================================================ */

function processPrediction(
    prediction
) {

    const smoothed =
        smoothPrediction(
            prediction
        );


    const isHelmet =
        smoothed.index === 0;


    let state =
        "verify";


    /*
        CASCO
    */

    if (
        isHelmet &&
        smoothed.confidence >=
        HELMET_MIN_CONFIDENCE
    ) {

        state =
            "helmet";
    }


    /*
        SIN CASCO
    */

    else if (
        !isHelmet &&
        smoothed.confidence >=
        VERIFY_CONFIDENCE
    ) {

        state =
            "nohelmet";
    }


    /*
        CONFIANZA BAJA
    */

    else {

        state =
            "verify";
    }


    updateInterface(
        state,
        smoothed.confidence
    );


    if (
        state !== lastState
    ) {

        playSoundForState(
            state
        );

        lastState =
            state;
    }
}


/* ============================================================
   INTERFAZ
============================================================ */

function updateInterface(
    state,
    confidence
) {

    const safeConfidence =
        clamp(
            confidence,
            0,
            1
        );


    confidenceValue.textContent =
        percent(
            safeConfidence
        );


    confidenceFill.style.width =
        `${safeConfidence * 100}%`;


    resultCard.className =
        `result-card state-${state}`;


    setHologramState(
        state
    );


    if (state === "helmet") {

        resultIcon.textContent =
            "✓";

        resultState.textContent =
            "CASCO";

        resultMessage.textContent =
            "Casco de seguridad detectado.";

        adviceText.textContent =
            "Mantén el casco correctamente colocado y visible.";

        cameraAdvice.textContent =
            "✓ CASCO DETECTADO";

        confidenceFill.style.background =
            "var(--green)";

    }


    else if (
        state === "nohelmet"
    ) {

        resultIcon.textContent =
            "!";

        resultState.textContent =
            "SIN CASCO";

        resultMessage.textContent =
            "No se detectó un casco de seguridad.";

        adviceText.textContent =
            "Colóquese el casco de seguridad antes de continuar.";

        cameraAdvice.textContent =
            "⚠ SIN CASCO DETECTADO";

        confidenceFill.style.background =
            "var(--red)";

    }


    else {

        resultIcon.textContent =
            "?";

        resultState.textContent =
            "VERIFICAR";

        resultMessage.textContent =
            "La IA no tiene suficiente confianza para determinar el estado.";

        adviceText.textContent =
            "Colócate de frente, mejora la iluminación y mantén el casco visible.";

        cameraAdvice.textContent =
            "⚠ VERIFICANDO";

        confidenceFill.style.background =
            "var(--yellow)";
    }
}


/* ============================================================
   HOLOGRAMA
============================================================ */

function setHologramState(
    state
) {

    hologram.classList.remove(
        "helmet",
        "nohelmet",
        "verify"
    );


    if (
        state === "helmet"
    ) {

        hologram.classList.add(
            "helmet"
        );

        targetIcon.textContent =
            "✓";

        trackingLabel.textContent =
            "ANÁLISIS: CASCO";

    }


    else if (
        state === "nohelmet"
    ) {

        hologram.classList.add(
            "nohelmet"
        );

        targetIcon.textContent =
            "!";

        trackingLabel.textContent =
            "ALERTA: SIN CASCO";

    }


    else {

        hologram.classList.add(
            "verify"
        );

        targetIcon.textContent =
            "?";

        trackingLabel.textContent =
            "ESCANEANDO";
    }
}


/* ============================================================
   FPS
============================================================ */

function updateFPS() {

    fpsFrames++;

    const now =
        performance.now();


    const elapsed =
        now - fpsLastTime;


    if (
        elapsed >= 1000
    ) {

        currentFPS =
            Math.round(
                fpsFrames *
                1000 /
                elapsed
            );


        fpsCounter.textContent =
            `${currentFPS} FPS`;


        fpsFrames = 0;

        fpsLastTime =
            now;
    }
}


/* ============================================================
   BOTONES
============================================================ */

startButton.addEventListener(
    "click",
    async () => {

        /*
            Crear el AudioContext después
            de una acción del usuario evita
            bloqueos de audio del navegador.
        */

        const audio =
            getAudioContext();


        if (
            audio &&
            audio.state ===
            "suspended"
        ) {

            try {
                await audio.resume();
            } catch (_) {}
        }


        if (!session) {

            try {

                await loadModel();

            } catch (error) {

                return;
            }
        }


        await startCamera();
    }
);


stopButton.addEventListener(
    "click",
    () => {

        stopCamera();
    }
);


/* ============================================================
   LIMPIEZA AL CERRAR
============================================================ */

window.addEventListener(
    "beforeunload",
    () => {

        stopCamera();
    }
);


/* ============================================================
   INICIALIZACIÓN
============================================================ */

async function initialize() {

    /*
        No encendemos la cámara automáticamente.
        El navegador necesita interacción del usuario.
    */

    cameraStatus.textContent =
        "CÁMARA INACTIVA";


    liveIndicator.innerHTML =
        "<span></span> OFFLINE";


    try {

        await loadModel();

    } catch (error) {

        console.error(
            "Inicialización fallida:",
            error
        );
    }
}


initialize();