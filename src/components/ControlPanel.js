import { formatNumber } from '../utils.js';

export function renderControls(state) {
  const disabled = state.isTraining ? 'disabled' : '';
  const lrExponent = Math.log10(state.learningRate);
  return `<section class="control-card card" aria-labelledby="control-title">
    <h2 id="control-title">Model controls</h2>
    <div class="control"><label class="control-label" for="degree-control">Polynomial degree <output for="degree-control">${state.degree}</output></label><input id="degree-control" data-control="degree" type="range" min="0" max="20" step="1" value="${state.degree}" ${disabled}></div>
    <div class="control"><label class="control-label" for="learning-rate-control">Learning rate <output for="learning-rate-control">${formatNumber(state.learningRate, 4)}</output></label><input id="learning-rate-control" data-control="learningRate" type="range" min="-4" max="0" step="0.01" value="${lrExponent}" ${disabled}></div>
    <div class="control"><label class="control-label" for="lambda-control">Regularization λ <output for="lambda-control">${state.lambda.toFixed(2)}</output></label><input id="lambda-control" data-control="lambda" type="range" min="0" max="1" step="0.01" value="${state.lambda}" ${disabled}></div>
    <div class="control"><label class="control-label" for="noise-control">Noise level <output for="noise-control">${state.noiseStd.toFixed(2)}</output></label><input id="noise-control" data-control="noiseStd" type="range" min="0" max="1" step="0.01" value="${state.noiseStd}" ${disabled}></div>
    <div class="control"><label class="control-label" for="points-control">Data points <output for="points-control">${state.numPoints}</output></label><input id="points-control" data-control="numPoints" type="range" min="50" max="2000" step="10" value="${state.numPoints}" ${disabled}></div>
    <div class="actions three"><button class="button" id="generate" ${disabled}>New data</button><button class="button" id="retrain" ${disabled}>Reset + retrain</button><button class="button primary" id="toggle">${state.isTraining ? 'Pause' : 'Start'}</button></div>
    <div class="status"><span>epoch</span><strong>${state.epochs} / ${state.maxEpochs}</strong></div>
  </section>`;
}
