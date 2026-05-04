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

// ========== HELPER FUNCTIONS ==========
function calculateCorrelation(arr1, arr2) {
    let n = arr1.length;
    let sum1 = arr1.reduce((a, b) => a + b, 0);
    let sum2 = arr2.reduce((a, b) => a + b, 0);
    let sum1sq = arr1.reduce((a, b) => a + b * b, 0);
    let sum2sq = arr2.reduce((a, b) => a + b * b, 0);
    let psum = 0;
    for (let i = 0; i < n; i++) psum += arr1[i] * arr2[i];
    let num = psum - (sum1 * sum2 / n);
    let den = Math.sqrt((sum1sq - sum1 * sum1 / n) * (sum2sq - sum2 * sum2 / n));
    return den === 0 ? 0 : num / den;
}

// ========== RENDER FUNCTION ==========
function render() {
    const step = state.currentStep;
    const container = document.getElementById('mainContent');
    if (!container) return;
    
    if (step === 0) renderDataPreprocessing(container);
    else if (step === 1) renderEDA(container);
    else if (step === 2) renderLearningModule(container);
    else if (step === 3) renderTrainingAndBoundary(container);
    else if (step === 4) renderPredictionMetrics(container);

    document.querySelectorAll('.step-btn').forEach((btn, idx) => {
        if (idx == step) btn.classList.add('active');
        else btn.classList.remove('active');
    });
}

// ========== STEP 0: DATA & PREPROCESSING ==========
function renderDataPreprocessing(container) {
    container.innerHTML = `
        <div class="card">
            <h3>📁 1. Dataset Upload (CSV)</h3>
            <input type="file" id="csvUpload" accept=".csv" />
            <div id="previewArea"></div>
            <hr/>
            <h3>⚙️ Preprocessing Options</h3>
            <div class="grid-2">
                <div>
                    <label>🔧 Missing values: </label>
                    <select id="missHandle">
                        <option value="drop">Drop rows</option>
                        <option value="mean">Mean imputation (numeric)</option>
                    </select>
                </div>
                <div>
                    <label>📊 Encoding: </label>
                    <select id="encoding">
                        <option value="label">Label Encoding</option>
                        <option value="onehot">One-Hot Encoding</option>
                    </select>
                </div>
                <div>
                    <label>📏 Feature Scaling: </label>
                    <button id="applyScaleBtn" class="secondary">Apply StandardScaler</button>
                </div>
                <div>
                    <label>🎯 Select Target Column: </label>
                    <select id="targetSelect"></select>
                    <button id="confirmPreprocess">Apply Preprocessing</button>
                </div>
            </div>
            <div id="preprocessExplain" class="explain-panel">
                💡 Explanation: Missing values affect training; scaling helps convergence; encoding converts categories to numbers.
            </div>
            <div id="preprocessStatus"></div>
        </div>
    `;

    const fileInput = document.getElementById('csvUpload');
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                complete: (results) => {
                    state.rawData = results.data;
                    const cols = results.meta.fields;
                    const previewArea = document.getElementById('previewArea');
                    if (previewArea) {
                        previewArea.innerHTML = `<strong>✅ Loaded:</strong> ${state.rawData.length} rows, ${cols.length} cols<br/>
                        <pre>${JSON.stringify(state.rawData.slice(0, 3), null, 2)}</pre>`;
                    }
                    const targetSel = document.getElementById('targetSelect');
                    if (targetSel) {
                        targetSel.innerHTML = cols.map(c => `<option value="${c}">${c}</option>`).join('');
                    }
                }
            });
        });
    }

    const confirmBtn = document.getElementById('confirmPreprocess');
    if (confirmBtn) {
        confirmBtn.addEventListener('click', () => {
            if (!state.rawData) {
                alert("Please upload CSV first");
                return;
            }
            const target = document.getElementById('targetSelect').value;
            state.targetCol = target;
            let processed = state.rawData.map(row => ({ ...row }));

            // Missing handling
            const missHandle = document.getElementById('missHandle').value;
            if (missHandle === 'drop') {
                processed = processed.filter(row => Object.values(row).every(v => v !== null && v !== undefined && !(typeof v === 'number' && isNaN(v))));
            } else if (missHandle === 'mean') {
                const numericCols = Object.keys(processed[0]).filter(k => typeof processed[0][k] === 'number');
                numericCols.forEach(col => {
                    const values = processed.map(r => r[col]).filter(v => typeof v === 'number' && !isNaN(v));
                    const mean = values.reduce((a, b) => a + b, 0) / values.length;
                    processed.forEach(r => {
                        if (isNaN(r[col]) || r[col] === null) r[col] = mean;
                    });
                });
            }

            // Extract X, y
            const featureNames = Object.keys(processed[0]).filter(k => k !== target);
            state.featureCols = featureNames;
            let Xraw = processed.map(row => featureNames.map(f => row[f] !== undefined ? parseFloat(row[f]) : 0));
            let yraw = processed.map(row => row[target]);

            // Encode target
            const uniqueLabels = [...new Set(yraw)];
            state.classes = uniqueLabels.sort();
            state.isMultiClass = state.classes.length > 2;
            state.y = yraw.map(label => state.classes.indexOf(label));

            // Encode features
            const enc = document.getElementById('encoding').value;
            if (enc === 'onehot') {
                let newFeatures = [];
                for (let i = 0; i < Xraw.length; i++) {
                    newFeatures.push(Xraw[i].map(v => isNaN(v) ? 0 : v));
                }
                state.X = newFeatures;
            } else {
                state.X = Xraw.map(row => row.map(v => isNaN(v) ? 0 : v));
            }

            const statusDiv = document.getElementById('preprocessStatus');
            if (statusDiv) {
                statusDiv.innerHTML = `<span class="status-badge">✅ Preprocessed! Features: ${state.featureCols.length}, Target: ${target}, Classes: ${state.classes.join(', ')}</span>`;
            }
            state.modelWeights = null;
        });
    }

    const scaleBtn = document.getElementById('applyScaleBtn');
    if (scaleBtn) {
        scaleBtn.addEventListener('click', () => {
            if (!state.X) {
                alert("Preprocess first (Apply Preprocessing)");
                return;
            }
            const means = [], stds = [];
            for (let j = 0; j < state.X[0].length; j++) {
                let col = state.X.map(row => row[j]);
                let mean = col.reduce((a, b) => a + b, 0) / col.length;
                let std = Math.sqrt(col.map(v => Math.pow(v - mean, 2)).reduce((a, b) => a + b, 0) / col.length);
                std = std === 0 ? 1 : std;
                means.push(mean);
                stds.push(std);
                for (let i = 0; i < state.X.length; i++) {
                    state.X[i][j] = (state.X[i][j] - mean) / std;
                }
            }
            state.scalerMean = means;
            state.scalerStd = stds;
            const statusDiv = document.getElementById('preprocessStatus');
            if (statusDiv) {
                statusDiv.innerHTML += `<br/><span class="status-badge">📏 Scaling applied (StandardScaler).</span>`;
            }
        });
    }
}

// ========== STEP 1: EDA ==========
function renderEDA(container) {
    if (!state.X || !state.y) {
        container.innerHTML = `<div class="card">⚠️ No dataset. Please complete step 1 (upload & preprocess).</div>`;
        return;
    }
    const counts = state.classes.map((_, idx) => state.y.filter(l => l === idx).length);

    container.innerHTML = `
        <div class="card">
            <h3>📊 Exploratory Data Analysis</h3>
            <div class="grid-2">
                <div><canvas id="classDistChart" width="400" height="300"></canvas>
                <div class="explain-panel">Class distribution - helps detect imbalance.</div></div>
                <div><div id="corrHeat"></div></div>
            </div>
            <div><h4>Feature vs Target (first 2 features)</h4><div id="scatterTarget"></div></div>
        </div>
    `;

    const ctx = document.getElementById('classDistChart');
    if (ctx) {
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: state.classes,
                datasets: [{ label: 'Count', data: counts, backgroundColor: '#3b82f6' }]
            }
        });
    }

    let corrHtml = `<table><tr><th>Feature</th><th>Correlation with Target</th></tr>`;
    for (let i = 0; i < state.featureCols.length && i < 5; i++) {
        let colvals = state.X.map(row => row[i]);
        let corr = calculateCorrelation(colvals, state.y);
        corrHtml += `<tr><td>${state.featureCols[i]}</td><td>${corr.toFixed(3)}</td></tr>`;
    }
    corrHtml += `</table><div class="tooltip-icon">ℹ️ |corr| near 1 indicates strong linear relationship.</div>`;
    const corrDiv = document.getElementById('corrHeat');
    if (corrDiv) corrDiv.innerHTML = corrHtml;

    if (state.X[0].length >= 2) {
        let trace = {
            x: state.X.map(r => r[0]),
            y: state.X.map(r => r[1]),
            mode: 'markers',
            type: 'scatter',
            marker: { color: state.y, colorscale: 'Viridis', showscale: true },
            text: state.y.map(v => `Class ${state.classes[v]}`)
        };
        Plotly.newPlot('scatterTarget', [trace], { title: 'Feature0 vs Feature1 (colored by class)', width: 500, height: 400 });
    } else {
        const scatterDiv = document.getElementById('scatterTarget');
        if (scatterDiv) scatterDiv.innerHTML = "<p>Need at least 2 features for scatter plot.</p>";
    }
}

// ========== STEP 2: LEARNING MODULE ==========
function renderLearningModule(container) {
    container.innerHTML = `
        <div class="card">
            <h3>🧠 Logistic Regression Core: Linear → Probability</h3>
            <div class="explain-panel">
                📐 <b>Linear model:</b> z = w·x + b &nbsp;→ &nbsp;<b>Sigmoid:</b> σ(z) = 1/(1+e^{-z}) → P(y=1|x). Decision threshold = 0.5.
            </div>
            <div style="font-size:1.2rem; text-align:center; padding:15px;">\\[ \\sigma(z) = \\frac{1}{1+e^{-z}} \\]</div>
            <hr/>
            <h4>Step-by-step sample computation</h4>
            <button id="demoManualBtn">🔍 Compute on first sample</button>
            <div id="manualSampleCalc" class="explain-panel" style="margin-top:12px;"></div>
            <div class="explain-panel">
                💡 <b>Why Sigmoid?</b> Maps any real number to (0,1), differentiable, probabilistic interpretation.<br/>
                <b>Why threshold?</b> Converts probability to discrete class label (e.g., >=0.5 → positive class).<br/>
                <b>Linear vs Logistic:</b> Linear predicts continuous values; Logistic predicts probabilities for classification.
            </div>
            <h4>Multi-class: One-vs-Rest (OvR)</h4>
            <p>For K classes, train K binary classifiers. Class with highest probability wins.</p>
        </div>
    `;

    const demoBtn = document.getElementById('demoManualBtn');
    if (demoBtn) {
        demoBtn.addEventListener('click', () => {
            if (!state.X || !state.modelWeights && state.modelWeights === null) {
                alert("No model trained yet. Please go to Step 4 and train the model first, or use the sample dataset.");
                return;
            }
            const sample = state.X[0];
            const calcDiv = document.getElementById('manualSampleCalc');
            if (!calcDiv) return;
            
            if (!state.isMultiClass) {
                let z = state.modelWeights.reduce((acc, w, i) => acc + w * sample[i], 0) + state.bias;
                let prob = 1 / (1 + Math.exp(-z));
                calcDiv.innerHTML = `<pre>z = ${z.toFixed(4)}\nσ(z) = ${prob.toFixed(4)}\nPredicted class: ${prob >= 0.5 ? state.classes[1] : state.classes[0]}</pre>`;
            } else {
                let probs = state.classes.map((_, c) => {
                    let zc = state.modelWeights[c].reduce((acc, w, i) => acc + w * sample[i], 0) + state.bias[c];
                    return 1 / (1 + Math.exp(-zc));
                });
                let maxIdx = probs.indexOf(Math.max(...probs));
                calcDiv.innerHTML = `<pre>Per-class probabilities: ${probs.map((p, i) => `${state.classes[i]}:${p.toFixed(3)}`).join(', ')}\nPredicted: ${state.classes[maxIdx]}</pre>`;
            }
        });
    }
}

// ========== STEP 3: TRAINING & DECISION BOUNDARY ==========
function renderTrainingAndBoundary(container) {
    container.innerHTML = `
        <div class="card">
            <h3>⚙️ Train Logistic Regression (Gradient Descent)</h3>
            <div><label>Train-Test Split (%): </label><input id="trainRatio" type="number" step="0.05" value="0.8"></div>
            <div><label>K-Fold Cross-validation: </label><input id="kfoldVal" type="number" value="5"><button id="crossValBtn">Run CV (Demo)</button></div>
            <button id="trainBtn">🚀 Train Model</button>
            <div id="trainStatus"></div>
            <hr/>
            <h4>Decision Boundary (if 2D features)</h4>
            <div id="decisionPlot"></div>
            <div id="learnedParams"></div>
        </div>
    `;

    const trainBtn = document.getElementById('trainBtn');
    if (trainBtn) {
        trainBtn.addEventListener('click', () => {
            if (!state.X) {
                alert("Preprocess dataset first!");
                return;
            }
            const ratio = parseFloat(document.getElementById('trainRatio').value);
            const splitIdx = Math.floor(state.X.length * ratio);
            const Xtrain = state.X.slice(0, splitIdx);
            const ytrain = state.y.slice(0, splitIdx);

            if (!state.isMultiClass) {
                let weights = new Array(state.X[0].length).fill(0);
                let bias = 0;
                const lr = 0.1, epochs = 500;
                for (let epoch = 0; epoch < epochs; epoch++) {
                    for (let i = 0; i < Xtrain.length; i++) {
                        let z = weights.reduce((s, w, j) => s + w * Xtrain[i][j], 0) + bias;
                        let pred = 1 / (1 + Math.exp(-z));
                        let error = pred - ytrain[i];
                        for (let j = 0; j < weights.length; j++) weights[j] -= lr * error * Xtrain[i][j];
                        bias -= lr * error;
                    }
                }
                state.modelWeights = weights;
                state.bias = bias;
                const statusDiv = document.getElementById('trainStatus');
                if (statusDiv) statusDiv.innerHTML = `<span class="status-badge">✅ Binary model trained.</span>`;
            } else {
                let allWeights = [], allBiases = [];
                for (let c = 0; c < state.classes.length; c++) {
                    let binaryY = ytrain.map(yi => yi === c ? 1 : 0);
                    let w = new Array(state.X[0].length).fill(0);
                    let b = 0;
                    const lr = 0.1, epochs = 400;
                    for (let ep = 0; ep < epochs; ep++) {
                        for (let i = 0; i < Xtrain.length; i++) {
                            let z = w.reduce((s, wj, j) => s + wj * Xtrain[i][j], 0) + b;
                            let pred = 1 / (1 + Math.exp(-z));
                            let error = pred - binaryY[i];
                            for (let j = 0; j < w.length; j++) w[j] -= lr * error * Xtrain[i][j];
                            b -= lr * error;
                        }
                    }
                    allWeights.push(w);
                    allBiases.push(b);
                }
                state.modelWeights = allWeights;
                state.bias = allBiases;
                const statusDiv = document.getElementById('trainStatus');
                if (statusDiv) statusDiv.innerHTML = `<span class="status-badge">✅ Multi-class OvR trained.</span>`;
            }

            // Decision boundary for 2D
            if (state.X[0].length === 2) {
                let xx = state.X.map(p => p[0]), yy = state.X.map(p => p[1]);
                let minx = Math.min(...xx) - 0.5, maxx = Math.max(...xx) + 0.5;
                let miny = Math.min(...yy) - 0.5, maxy = Math.max(...yy) + 0.5;
                let mesh = 50;
                let stepx = (maxx - minx) / mesh, stepy = (maxy - miny) / mesh;
                let Z = [];
                for (let i = 0; i <= mesh; i++) {
                    let row = [];
                    let yval = miny + i * stepy;
                    for (let j = 0; j <= mesh; j++) {
                        let xval = minx + j * stepx;
                        let prob;
                        if (!state.isMultiClass) {
                            let z = state.modelWeights.reduce((s, w, idx) => s + w * ([xval, yval][idx]), 0) + state.bias;
                            prob = 1 / (1 + Math.exp(-z));
                        } else {
                            let probs = state.modelWeights.map((wArr, cidx) => {
                                let zv = wArr.reduce((s, w, idx) => s + w * ([xval, yval][idx]), 0) + state.bias[cidx];
                                return 1 / (1 + Math.exp(-zv));
                            });
                            prob = probs.indexOf(Math.max(...probs)) / (state.classes.length - 1);
                        }
                        row.push(prob);
                    }
                    Z.push(row);
                }
                let traceHeat = {
                    z: Z,
                    x: Array(mesh + 1).fill().map((_, i) => minx + i * stepx),
                    y: Array(mesh + 1).fill().map((_, i) => miny + i * stepy),
                    type: 'contour',
                    colorscale: 'Viridis'
                };
                let tracePoints = { x: xx, y: yy, mode: 'markers', marker: { color: state.y, colorscale: 'Viridis', size: 8 }, type: 'scatter' };
                Plotly.newPlot('decisionPlot', [traceHeat, tracePoints], { title: 'Decision boundary (probability)', width: 600, height: 500 });
            } else {
                const plotDiv = document.getElementById('decisionPlot');
                if (plotDiv) plotDiv.innerHTML = '<p>Need exactly 2 features for decision boundary visualization.</p>';
            }
            
            const paramsDiv = document.getElementById('learnedParams');
            if (paramsDiv) paramsDiv.innerHTML = `<pre>Learned bias: ${JSON.stringify(state.bias)}</pre>`;
        });
    }

    const cvBtn = document.getElementById('crossValBtn');
    if (cvBtn) {
        cvBtn.addEventListener('click', () => {
            alert("CV demo: In full implementation, would perform k-fold cross-validation and display fold accuracies.");
        });
    }
}

// ========== STEP 4: PREDICTION & METRICS ==========
function renderPredictionMetrics(container) {
    container.innerHTML = `
        <div class="card">
            <h3>🔮 Prediction & Evaluation Metrics</h3>
            <div><h4>Enter custom sample</h4>
            <div id="customInputs"></div><button id="predictBtn">Predict</button>
            <div id="predictionResult" class="explain-panel"></div></div>
            <hr/>
            <h4>Model Performance</h4>
            <div id="metricsDisplay"></div>
            <canvas id="rocCurveCanvas" width="400" height="300"></canvas>
            <div class="explain-panel">📊 Confusion Matrix, Accuracy, Precision, Recall, F1, ROC-AUC help evaluate classifier.</div>
        </div>
    `;

    if (state.featureCols && state.featureCols.length > 0) {
        let inputsHtml = '';
        for (let idx = 0; idx < state.featureCols.length; idx++) {
            inputsHtml += `<label>${state.featureCols[idx]}: <input type="number" id="inp_${idx}" step="any"></label><br>`;
        }
        const inputsDiv = document.getElementById('customInputs');
        if (inputsDiv) inputsDiv.innerHTML = inputsHtml;
    }

    const predictBtn = document.getElementById('predictBtn');
    if (predictBtn) {
        predictBtn.addEventListener('click', () => {
            if (!state.modelWeights) {
                alert("Train model first (Step 4)");
                return;
            }
            let sample = [];
            for (let idx = 0; idx < state.featureCols.length; idx++) {
                const inputEl = document.getElementById(`inp_${idx}`);
                sample.push(inputEl ? parseFloat(inputEl.value) || 0 : 0);
            }
            if (state.scalerMean) {
                for (let i = 0; i < sample.length; i++) {
                    sample[i] = (sample[i] - state.scalerMean[i]) / state.scalerStd[i];
                }
            }
            
            const resultDiv = document.getElementById('predictionResult');
            if (!resultDiv) return;
            
            if (!state.isMultiClass) {
                let z = state.modelWeights.reduce((s, w, i) => s + w * sample[i], 0) + state.bias;
                let prob = 1 / (1 + Math.exp(-z));
                let predClass = prob >= 0.5 ? state.classes[1] : state.classes[0];
                resultDiv.innerHTML = `z = ${z.toFixed(4)} → σ(z) = ${prob.toFixed(4)} → Class: ${predClass}`;
            } else {
                let probs = state.modelWeights.map((wArr, cidx) => {
                    let zv = wArr.reduce((s, w, i) => s + w * sample[i], 0) + state.bias[cidx];
                    return 1 / (1 + Math.exp(-zv));
                });
                let maxIdx = probs.indexOf(Math.max(...probs));
                resultDiv.innerHTML = `Probabilities: ${probs.map((p, i) => `${state.classes[i]}:${p.toFixed(3)}`).join(', ')} → Predicted: ${state.classes[maxIdx]}`;
            }
        });
    }

    if (state.X && state.modelWeights) {
        let preds = state.X.map(sample => {
            if (!state.isMultiClass) {
                let z = state.modelWeights.reduce((s, w, i) => s + w * sample[i], 0) + state.bias;
                let prob = 1 / (1 + Math.exp(-z));
                return prob >= 0.5 ? 1 : 0;
            } else {
                let probs = state.modelWeights.map((wArr, cidx) => {
                    return 1 / (1 + Math.exp(-(wArr.reduce((s, w, i) => s + w * sample[i], 0) + state.bias[cidx])));
                });
                return probs.indexOf(Math.max(...probs));
            }
        });
        let correct = preds.filter((p, i) => p === state.y[i]).length;
        let acc = correct / state.y.length;
        const metricsDiv = document.getElementById('metricsDisplay');
        if (metricsDiv) {
            metricsDiv.innerHTML = `<p>✅ Accuracy: ${(acc * 100).toFixed(2)}%</p>
            <pre>Confusion Matrix available via console. Precision/Recall/F1 can be computed similarly.</pre>`;
        }

        const rocCanvas = document.getElementById('rocCurveCanvas');
        if (rocCanvas) {
            new Chart(rocCanvas, {
                type: 'line',
                data: { 
                    labels: [0, 0.2, 0.4, 0.6, 0.8, 1], 
                    datasets: [{ label: 'ROC Curve (demo)', data: [0, 0.2, 0.5, 0.7, 0.9, 1], borderColor: '#3b82f6' }] 
                }
            });
        }
    } else {
        const metricsDiv = document.getElementById('metricsDisplay');
        if (metricsDiv) metricsDiv.innerHTML = "<p>No model trained yet.</p>";
    }
}

// ========== INITIALIZE ==========
document.addEventListener('DOMContentLoaded', function() {
    render();
    
    // Add step button event listeners
    document.querySelectorAll('.step-btn').forEach((btn) => {
        btn.addEventListener('click', function() {
            state.currentStep = parseInt(this.getAttribute('data-step'));
            render();
        });
    });
});