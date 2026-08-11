import './style.css';
import { generateDataset, splitData, X_MIN, X_MAX } from './data.js';
import { initializeWeights, predict, trainEpoch } from './model.js';

const state = { numPoints: 300, noiseStd: 0.3, degree: 3, learningRate: 0.03, lambda: 0, batchSize: 32, epochs: 0, maxEpochs: 500, weights: [], trainingLosses: [], validationLosses: [], trainX: [], trainY: [], valX: [], valY: [], isTraining: false, trainingComplete: false, divergenceDetected: false, overfittingWarning: false, tab: 'playground' };
let timer;

const app = document.querySelector('#app');
app.innerHTML = `<div class="shell"><header class="topbar"><div class="brand"><span class="brand-mark">∑</span><span>POLY FIT / LAB 01</span></div><a class="github" href="https://github.com/isaiahcampusano/ml-regression" target="_blank" rel="noreferrer">View on GitHub ↗</a></header><div class="hero"><div><p class="eyebrow">A visual field guide to model complexity</p><h1>Overfitting,<br><em>in action.</em></h1></div><p class="hero-copy">Explore how a polynomial learns a noisy signal — and watch validation loss reveal when the model learns too much.</p></div><nav class="tabs" aria-label="App sections"><button class="tab active" data-tab="playground">Playground</button><button class="tab" data-tab="learn">Learn</button></nav><main id="content"></main></div>`;

function setupData() {
  const dataset = generateDataset(state.numPoints, state.noiseStd);
  const split = splitData(dataset.x, dataset.y);
  Object.assign(state, split, { epochs: 0, weights: initializeWeights(state.degree), trainingLosses: [], validationLosses: [], trainingComplete: false, divergenceDetected: false, overfittingWarning: false });
}

function esc(value) { return String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[char])); }
function nice(value) { return Number.isFinite(value) ? value.toFixed(3) : '—'; }
function pathFor(points, xScale, yScale) { return points.map((point, index) => `${index ? 'L' : 'M'} ${xScale(point[0]).toFixed(1)} ${yScale(point[1]).toFixed(1)}`).join(' '); }

function chartFrame(width, height, margin, xTicks, yTicks, xScale, yScale, xLabel, yLabel) {
  let markup = '';
  yTicks.forEach((tick) => { markup += `<line class="grid" x1="${margin.left}" x2="${width - margin.right}" y1="${yScale(tick)}" y2="${yScale(tick)}"/><text class="axis-label" x="${margin.left - 9}" y="${yScale(tick) + 3}" text-anchor="end">${nice(tick)}</text>`; });
  xTicks.forEach((tick) => { markup += `<line class="grid" x1="${xScale(tick)}" x2="${xScale(tick)}" y1="${margin.top}" y2="${height - margin.bottom}"/><text class="axis-label" x="${xScale(tick)}" y="${height - margin.bottom + 18}" text-anchor="middle">${nice(tick)}</text>`; });
  markup += `<line class="axis" x1="${margin.left}" x2="${width - margin.right}" y1="${height - margin.bottom}" y2="${height - margin.bottom}"/><line class="axis" x1="${margin.left}" x2="${margin.left}" y1="${margin.top}" y2="${height - margin.bottom}"/><text class="axis-label" x="${width / 2}" y="${height - 3}" text-anchor="middle">${xLabel}</text><text class="axis-label" transform="translate(11 ${height / 2}) rotate(-90)" text-anchor="middle">${yLabel}</text>`;
  return markup;
}

function scatterSvg() {
  const width = 760, height = 410, margin = { top: 14, right: 18, bottom: 35, left: 44 };
  const allY = [...state.trainY, ...state.valY];
  const curve = Array.from({ length: 180 }, (_, i) => { const x = X_MIN + (X_MAX - X_MIN) * i / 179; return [x, predict(state.weights, x)]; });
  const minY = Math.min(...allY, ...curve.map((point) => point[1])) - .18;
  const maxY = Math.max(...allY, ...curve.map((point) => point[1])) + .18;
  const x = (value) => margin.left + (value - X_MIN) / (X_MAX - X_MIN) * (width - margin.left - margin.right);
  const y = (value) => height - margin.bottom - (value - minY) / (maxY - minY || 1) * (height - margin.top - margin.bottom);
  const yTicks = Array.from({ length: 5 }, (_, i) => minY + (maxY - minY) * i / 4);
  const marks = chartFrame(width, height, margin, [-3, -1.5, 0, 1.5, 3], yTicks, x, y, 'input x', 'output y');
  const train = state.trainX.map((value, index) => `<circle cx="${x(value)}" cy="${y(state.trainY[index])}" r="3.3" fill="#4d8e89" opacity=".64"/>`).join('');
  const validation = state.valX.map((value, index) => `<circle cx="${x(value)}" cy="${y(state.valY[index])}" r="3.4" fill="#fffdf8" stroke="#df6b4d" stroke-width="1.5" opacity=".86"/>`).join('');
  return `<svg class="scatter" viewBox="0 0 ${width} ${height}" role="img" aria-label="Scatter plot of train and validation data with fitted polynomial">${marks}<path d="${pathFor(curve, x, y)}" fill="none" stroke="#df6b4d" stroke-width="3" stroke-linecap="round"/>${train}${validation}</svg>`;
}

function lossSvg() {
  const width = 760, height = 180, margin = { top: 12, right: 18, bottom: 30, left: 44 };
  const maxEpoch = Math.max(10, state.epochs);
  const values = [...state.trainingLosses, ...state.validationLosses].filter(Number.isFinite);
  const maxLoss = Math.max(.1, ...(values.length ? values : [.1])) * 1.08;
  const x = (value) => margin.left + value / maxEpoch * (width - margin.left - margin.right);
  const y = (value) => height - margin.bottom - value / maxLoss * (height - margin.top - margin.bottom);
  const train = state.trainingLosses.map((value, index) => [index + 1, value]);
  const validation = state.validationLosses.map((value, index) => [index + 1, value]);
  const marks = chartFrame(width, height, margin, [0, Math.round(maxEpoch / 2), maxEpoch], [0, maxLoss / 2, maxLoss], x, y, 'epochs', 'loss');
  return `<svg class="loss" viewBox="0 0 ${width} ${height}" role="img" aria-label="Training and validation loss over epochs">${marks}${state.overfittingWarning ? `<rect x="${x(Math.max(1, state.epochs - 5))}" y="${margin.top}" width="${width - margin.right - x(Math.max(1, state.epochs - 5))}" height="${height - margin.top - margin.bottom}" fill="#df6b4d" opacity=".06"/>` : ''}<path d="${pathFor(train, x, y)}" fill="none" stroke="#4d8e89" stroke-width="2.5"/><path d="${pathFor(validation, x, y)}" fill="none" stroke="#df6b4d" stroke-width="2.5"/></svg>`;
}

function weightsHtml() {
  const sorted = [...state.weights].map((value, index) => ({ value, index })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 5);
  return `<div class="weight-card card"><h2>Largest weights</h2><table class="weights"><thead><tr><th>term</th><th>value</th></tr></thead><tbody>${sorted.map(({ value, index }) => `<tr class="${Math.abs(value) > 5 ? 'hot' : ''}"><td>w<sub>${index}</sub></td><td>${nice(value)}</td></tr>`).join('')}</tbody></table><p class="weight-note">${sorted[0] && Math.abs(sorted[0].value) > 5 ? 'Large weights can signal a curve that is chasing noise.' : 'Balanced weights usually mean the curve is capturing the broader signal.'}</p></div>`;
}

function controlsHtml() {
  const disabled = state.isTraining ? 'disabled' : '';
  return `<section class="control-card card"><h2>Model controls</h2><label class="control"><span class="control-label">Polynomial degree <span class="value">${state.degree}</span></span><input data-control="degree" type="range" min="0" max="20" step="1" value="${state.degree}" ${disabled}></label><label class="control"><span class="control-label">Learning rate <span class="value">${state.learningRate.toFixed(3)}</span></span><input data-control="learningRate" type="range" min="0.001" max="0.2" step="0.001" value="${state.learningRate}" ${disabled}></label><label class="control"><span class="control-label">Regularization λ <span class="value">${state.lambda.toFixed(2)}</span></span><input data-control="lambda" type="range" min="0" max="1" step="0.01" value="${state.lambda}" ${disabled}></label><label class="control"><span class="control-label">Noise level <span class="value">${state.noiseStd.toFixed(2)}</span></span><input data-control="noiseStd" type="range" min="0" max="1" step="0.01" value="${state.noiseStd}" ${disabled}></label><label class="control"><span class="control-label">Data points <span class="value">${state.numPoints}</span></span><input data-control="numPoints" type="range" min="50" max="1000" step="10" value="${state.numPoints}" ${disabled}></label><div class="actions"><button class="button" id="generate" ${disabled}>New dataset</button><button class="button primary" id="toggle">${state.isTraining ? 'Pause training' : state.trainingComplete ? 'Retrain' : 'Start training'}</button></div><div class="status"><span>epoch</span><strong>${state.epochs} / ${state.maxEpochs}</strong></div></section>${weightsHtml()}`;
}

function renderPlayground() {
  const alert = state.divergenceDetected ? '<div class="alert warning" role="alert">⚠ Training diverged. Try lowering the learning rate.</div>' : state.overfittingWarning ? '<div class="alert warning" role="alert">⚠ Validation loss is rising — overfitting is likely. Try more λ or a lower degree.</div>' : state.trainingComplete ? '<div class="alert" role="status">✓ Training complete. Compare the two loss lines to judge the fit.</div>' : '';
  return `${alert}<div class="layout"><div><section class="chart-card card"><div class="chart-head"><span class="chart-title">Signal + fitted curve</span><div class="legend"><span style="--dot:#4d8e89">train</span><span style="--dot:#df6b4d">validation</span><span style="--dot:#df6b4d">model</span></div></div>${scatterSvg()}</section><section class="chart-card card" style="margin-top:18px"><div class="chart-head"><span class="chart-title">Loss over time</span><div class="legend"><span style="--dot:#4d8e89">training loss</span><span style="--dot:#df6b4d">validation loss</span></div></div>${lossSvg()}</section></div><aside>${controlsHtml()}</aside></div>`;
}

function renderLearn() {
  return `<div class="learn"><div class="learn-grid"><section class="card"><h2>Polynomial regression</h2><p>A polynomial is a flexible curve made by adding powers of x. The degree controls how many bends the model can make.</p><div class="formula">ŷ = w₀ + w₁x + w₂x² + … + wᵈxᵈ</div></section><section class="card"><h2>Mini-batch SGD</h2><p>Stochastic gradient descent nudges each weight in the direction that lowers error. Small batches make each update quick and keep the process visible.</p><div class="formula">w ← w − η · gradient</div></section><section class="card"><h2>Train vs. validation</h2><p>We use 80% of the points to update the model and hold out 20% to test generalization. A useful model lowers both losses together.</p><ul><li>Training loss: error on points the model sees.</li><li>Validation loss: error on fresh held-out points.</li></ul></section><section class="card"><h2>The overfitting signal</h2><p>A high-degree curve can memorize noise. Training loss keeps falling, but validation loss turns upward. L2 regularization (λ) discourages giant weights and smooths the curve.</p><div class="formula">loss = MSE + λ⁄2 · Σ wₖ²</div></section></div></div>`;
}

function render() {
  document.querySelectorAll('.tab').forEach((tab) => tab.classList.toggle('active', tab.dataset.tab === state.tab));
  document.querySelector('#content').innerHTML = state.tab === 'learn' ? renderLearn() : renderPlayground();
  bindEvents();
}

function bindEvents() {
  document.querySelectorAll('[data-tab]').forEach((button) => button.addEventListener('click', () => { state.tab = button.dataset.tab; render(); }));
  document.querySelectorAll('[data-control]').forEach((input) => input.addEventListener('input', () => { state[input.dataset.control] = Number(input.value); setupData(); render(); }));
  document.querySelector('#generate')?.addEventListener('click', () => { setupData(); state.isTraining = true; render(); tick(); });
  document.querySelector('#toggle')?.addEventListener('click', () => { if (state.trainingComplete || state.divergenceDetected) setupData(); state.isTraining = !state.isTraining; render(); if (state.isTraining) tick(); else clearTimeout(timer); });
}

function tick() {
  if (!state.isTraining) return;
  trainEpoch(state);
  if (state.divergenceDetected || state.epochs >= state.maxEpochs) { state.isTraining = false; state.trainingComplete = !state.divergenceDetected; }
  render();
  if (state.isTraining) timer = setTimeout(tick, 22);
}

setupData();
render();
state.isTraining = true;
tick();
