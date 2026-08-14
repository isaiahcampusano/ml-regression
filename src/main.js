import './style.css';
import { generateDataset, splitData } from './data.js';
import { initializeWeights, trainEpoch } from './model.js';
import { appState, resetRun, setState } from './state.js';
import { renderTabs } from './components/TabContainer.js';
import { renderPlayground } from './components/PlaygroundTab.js';
import { renderLearn } from './components/LearnTab.js';

const app = document.querySelector('#app');
let timerId = null;

app.innerHTML = `
  <div class="shell">
    <header class="topbar">
      <a class="brand" href="#playground" aria-label="Poly Fit home">
        <span class="brand-mark">∑</span><span>POLY FIT / LAB 01</span>
      </a>
      <a class="github" href="https://github.com/isaiahcampusano/ml-regression" target="_blank" rel="noreferrer">View on GitHub ↗</a>
    </header>
    <section class="hero">
      <div><p class="eyebrow">A visual field guide to model complexity</p><h1>Overfitting,<br><em>in action.</em></h1></div>
      <p class="hero-copy">Explore how a polynomial learns a noisy signal — and watch validation loss reveal when the model learns too much.</p>
    </section>
    <nav id="tabs" class="tabs" aria-label="App sections"></nav>
    <main id="content" tabindex="-1"></main>
  </div>`;

const tabs = document.querySelector('#tabs');
const content = document.querySelector('#content');

function stopTimer() {
  if (timerId !== null) window.clearTimeout(timerId);
  timerId = null;
}

function createDataset() {
  const { x, y } = generateDataset(appState.numPoints, appState.noiseStd);
  Object.assign(appState, { rawX: x, rawY: y, ...splitData(x, y) });
  resetRun(initializeWeights(appState.degree));
}

function resetModel() {
  resetRun(initializeWeights(appState.degree));
}

function render() {
  tabs.innerHTML = renderTabs(appState.tab);
  content.innerHTML = appState.tab === 'learn' ? renderLearn() : renderPlayground(appState);
  bindEvents();
}

function startTraining() {
  if (!appState.trainX.length) createDataset();
  setState({ isTraining: true, trainingComplete: false, earlyStopped: false });
  render();
  tick();
}

function pauseTraining() {
  stopTimer();
  setState({ isTraining: false });
  render();
}

function tick() {
  if (!appState.isTraining) return;
  trainEpoch(appState);
  if (appState.divergenceDetected || appState.earlyStopped || appState.epochs >= appState.maxEpochs) {
    setState({ isTraining: false, trainingComplete: !appState.divergenceDetected });
  }
  render();
  if (appState.isTraining) timerId = window.setTimeout(tick, 24);
}

function bindEvents() {
  document.querySelectorAll('[data-tab]').forEach((button) => {
    button.addEventListener('click', () => {
      setState({ tab: button.dataset.tab });
      window.location.hash = button.dataset.tab;
      render();
    });
  });

  document.querySelectorAll('[data-control]').forEach((input) => {
    input.addEventListener('input', () => {
      const key = input.dataset.control;
      const value = key === 'learningRate' ? 10 ** Number(input.value) : Number(input.value);
      setState({ [key]: value });
      if (key === 'noiseStd' || key === 'numPoints') createDataset();
      else resetModel();
      render();
    });
  });

  document.querySelector('#generate')?.addEventListener('click', () => { createDataset(); startTraining(); });
  document.querySelector('#retrain')?.addEventListener('click', () => { resetModel(); startTraining(); });
  document.querySelector('#toggle')?.addEventListener('click', () => {
    if (appState.isTraining) pauseTraining();
    else {
      if (appState.trainingComplete || appState.divergenceDetected || appState.earlyStopped) resetModel();
      startTraining();
    }
  });
}

window.addEventListener('hashchange', () => {
  const tab = window.location.hash === '#learn' ? 'learn' : 'playground';
  if (tab !== appState.tab) { setState({ tab }); render(); }
});

appState.tab = window.location.hash === '#learn' ? 'learn' : 'playground';
createDataset();
render();
startTraining();
