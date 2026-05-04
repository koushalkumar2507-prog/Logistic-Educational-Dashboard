// ========== GLOBAL STATE ==========
let state = {
    rawData: null,
    featureCols: [],
    targetCol: null,
    X: null,
    y: null,
    classes: [],
    isMultiClass: false,
    modelWeights: null,
    bias: null,
    scalerMean: null,
    scalerStd: null,
    currentStep: 0
};

// ========== HELPER ==========
function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}

// ========== RENDER ==========
function render() {
    const container = document.getElementById('mainContent');
    if (!container) return;

    if (state.currentStep === 0) renderData(container);
    else if (state.currentStep === 1) renderEDA(container);
    else if (state.currentStep === 2) renderLearning(container);
    else if (state.currentStep === 3) renderTraining(container);
    else if (state.currentStep === 4) renderPrediction(container);
}

// ========== STEP 0 ==========
function renderData(container) {
    container.innerHTML = `
        <div class="card">
            <h3>Upload CSV</h3>
            <input type="file" id="csvUpload" />
            <select id="targetSelect"></select>
            <button id="applyPreprocess">Apply</button>
            <div id="status"></div>
        </div>
    `;

    document.getElementById('csvUpload').addEventListener('change', e => {
        const file = e.target.files[0];
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            complete: (res) => {
                state.rawData = res.data;
                const cols = res.meta.fields;

                document.getElementById('targetSelect').innerHTML =
                    cols.map(c => `<option>${c}</option>`).join('');
            }
        });
    });

    document.getElementById('applyPreprocess').addEventListener('click', () => {
        if (!state.rawData) return alert("Upload CSV first");

        const target = document.getElementById('targetSelect').value;
        state.targetCol = target;

        let data = state.rawData.filter(row =>
            Object.values(row).every(v => v !== null && v !== "")
        );

        const features = Object.keys(data[0]).filter(k => k !== target);
        state.featureCols = features;

        let X = data.map(row =>
            features.map(f => {
                let v = row[f];
                return typeof v === 'number' ? v : 0;
            })
        );

        let yRaw = data.map(r => r[target]);
        state.classes = [...new Set(yRaw)];
        state.y = yRaw.map(v => state.classes.indexOf(v));
        state.isMultiClass = state.classes.length > 2;

        state.X = X;

        document.getElementById('status').innerHTML =
            `Loaded ${X.length} samples, ${features.length} features`;
    });
}

// ========== STEP 1 ==========
function renderEDA(container) {
    if (!state.X) {
        container.innerHTML = "No data";
        return;
    }

    let counts = state.classes.map((_, i) =>
        state.y.filter(v => v === i).length
    );

    container.innerHTML = `
        <canvas id="chart"></canvas>
    `;

    new Chart(document.getElementById('chart'), {
        type: 'bar',
        data: {
            labels: state.classes,
            datasets: [{ data: counts }]
        }
    });
}

// ========== STEP 2 ==========
function renderLearning(container) {
    container.innerHTML = `
        <div>
            <h3>Sigmoid Function</h3>
            <p>σ(z) = 1 / (1 + e^-z)</p>
        </div>
    `;
}

// ========== STEP 3 ==========
function renderTraining(container) {
    container.innerHTML = `
        <button id="trainBtn">Train Model</button>
        <div id="trainStatus"></div>
    `;

    document.getElementById('trainBtn').onclick = () => {
        if (!state.X) return alert("No data");

        const split = Math.floor(0.8 * state.X.length);

        const Xtrain = state.X.slice(0, split);
        const ytrain = state.y.slice(0, split);
        const Xtest = state.X.slice(split);
        const ytest = state.y.slice(split);

        let weights = new Array(state.X[0].length).fill(0);
        let bias = 0;

        let lr = 0.1;
        let epochs = 300;

        for (let e = 0; e < epochs; e++) {
            for (let i = 0; i < Xtrain.length; i++) {
                let z = weights.reduce((s, w, j) => s + w * Xtrain[i][j], 0) + bias;
                let pred = sigmoid(z);
                let error = pred - ytrain[i];

                for (let j = 0; j < weights.length; j++) {
                    weights[j] -= lr * error * Xtrain[i][j];
                }
                bias -= lr * error;
            }
        }

        state.modelWeights = weights;
        state.bias = bias;

        // TEST ACCURACY
        let correct = 0;

        for (let i = 0; i < Xtest.length; i++) {
            let z = weights.reduce((s, w, j) => s + w * Xtest[i][j], 0) + bias;
            let pred = sigmoid(z) >= 0.5 ? 1 : 0;
            if (pred === ytest[i]) correct++;
        }

        let acc = correct / Xtest.length;

        document.getElementById('trainStatus').innerHTML =
            `✅ Trained | Test Accuracy: ${(acc * 100).toFixed(2)}%`;
    };
}

// ========== STEP 4 ==========
function renderPrediction(container) {
    if (!state.modelWeights) {
        container.innerHTML = "Train model first";
        return;
    }

    let inputs = state.featureCols.map((f, i) =>
        `<input id="f${i}" placeholder="${f}"/>`
    ).join("<br>");

    container.innerHTML = `
        ${inputs}
        <button id="predictBtn">Predict</button>
        <div id="result"></div>
    `;

    document.getElementById('predictBtn').onclick = () => {
        let sample = state.featureCols.map((_, i) =>
            parseFloat(document.getElementById(`f${i}`).value) || 0
        );

        let z = state.modelWeights.reduce((s, w, j) => s + w * sample[j], 0) + state.bias;
        let prob = sigmoid(z);

        let pred = prob >= 0.5 ? state.classes[1] : state.classes[0];

        document.getElementById('result').innerHTML =
            `Probability: ${prob.toFixed(3)} → ${pred}`;
    };
}

// ========== INIT ==========
document.addEventListener("DOMContentLoaded", () => {
    render();

    document.querySelectorAll('.step-btn').forEach(btn => {
        btn.onclick = () => {
            state.currentStep = parseInt(btn.dataset.step);
            render();
        };
    });
});
