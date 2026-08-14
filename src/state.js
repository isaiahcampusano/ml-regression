export const appState = {
  numPoints: 300,
  noiseStd: 0.3,
  xRange: [-3, 3],
  rawX: [], rawY: [], trainX: [], trainY: [], valX: [], valY: [],
  degree: 3,
  learningRate: 0.01,
  lambda: 0,
  batchSize: 32,
  weights: new Float64Array(),
  epochs: 0,
  maxEpochs: 500,
  trainingLosses: [],
  validationLosses: [],
  bestValidationLoss: Number.POSITIVE_INFINITY,
  epochsWithoutImprovement: 0,
  isTraining: false,
  trainingComplete: false,
  divergenceDetected: false,
  overfittingWarning: false,
  overfittingEpoch: null,
  earlyStopped: false,
  tab: 'playground',
};

export function setState(partial) {
  Object.assign(appState, partial);
  return appState;
}

export function resetRun(weights) {
  Object.assign(appState, {
    weights,
    epochs: 0,
    trainingLosses: [],
    validationLosses: [],
    bestValidationLoss: Number.POSITIVE_INFINITY,
    epochsWithoutImprovement: 0,
    isTraining: false,
    trainingComplete: false,
    divergenceDetected: false,
    overfittingWarning: false,
    overfittingEpoch: null,
    earlyStopped: false,
  });
}
