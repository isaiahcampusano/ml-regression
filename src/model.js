import { X_MAX } from './data.js';

export function polynomialFeatures(x, degree, scaleX = true) {
  const scaled = scaleX ? x / X_MAX : x;
  const values = new Float64Array(degree + 1);
  values[0] = 1;
  for (let index = 1; index <= degree; index += 1) values[index] = values[index - 1] * scaled;
  return values;
}

export const features = polynomialFeatures;

export function predict(weights, x) {
  const row = polynomialFeatures(x, weights.length - 1);
  let prediction = 0;
  for (let index = 0; index < weights.length; index += 1) prediction += weights[index] * row[index];
  return prediction;
}

export function computeLoss(xs, ys, weights, lambda = 0) {
  if (!xs.length) return 0;
  let squaredError = 0;
  for (let index = 0; index < xs.length; index += 1) {
    const error = predict(weights, xs[index]) - ys[index];
    squaredError += error * error;
  }
  let penalty = 0;
  for (let index = 1; index < weights.length; index += 1) penalty += weights[index] ** 2;
  return squaredError / xs.length + (lambda / 2) * penalty;
}

export const loss = computeLoss;

export function initializeWeights(degree, random = Math.random) {
  return Float64Array.from({ length: degree + 1 }, () => random() - 0.5);
}

export function sgdStep(xs, ys, weights, learningRate, lambda = 0) {
  if (!xs.length) return weights;
  const gradient = new Float64Array(weights.length);
  for (let rowIndex = 0; rowIndex < xs.length; rowIndex += 1) {
    const row = polynomialFeatures(xs[rowIndex], weights.length - 1);
    let prediction = 0;
    for (let index = 0; index < weights.length; index += 1) prediction += weights[index] * row[index];
    const error = prediction - ys[rowIndex];
    for (let index = 0; index < weights.length; index += 1) gradient[index] += (2 / xs.length) * error * row[index];
  }
  for (let index = 1; index < weights.length; index += 1) gradient[index] += lambda * weights[index];
  for (let index = 0; index < weights.length; index += 1) weights[index] -= learningRate * gradient[index];
  return weights;
}

function fisherYates(indices) {
  for (let index = indices.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [indices[index], indices[swapIndex]] = [indices[swapIndex], indices[index]];
  }
}

function hasOverfittingPattern(trainingLosses, validationLosses, windowSize = 5) {
  if (trainingLosses.length < 21 || validationLosses.length < windowSize + 1) return false;
  const start = validationLosses.length - windowSize - 1;
  const validationRising = validationLosses.slice(start).every((value, index, values) => index === 0 || value > values[index - 1]);
  const trainingFalling = trainingLosses.at(-1) < trainingLosses.at(-(windowSize + 1));
  return validationRising && trainingFalling;
}

export function trainEpoch(state, onUpdate = () => {}) {
  if (!state.trainX.length || !state.valX.length) return null;
  const order = state.trainX.map((_, index) => index);
  fisherYates(order);
  const batchSize = Math.max(1, Math.min(state.batchSize, order.length));
  const previousWeights = Float64Array.from(state.weights);

  for (let start = 0; start < order.length; start += batchSize) {
    const batch = order.slice(start, start + batchSize);
    sgdStep(
      batch.map((index) => state.trainX[index]),
      batch.map((index) => state.trainY[index]),
      state.weights,
      state.learningRate,
      state.lambda,
    );
  }

  const trainLoss = computeLoss(state.trainX, state.trainY, state.weights, state.lambda);
  const validationLoss = computeLoss(state.valX, state.valY, state.weights, state.lambda);
  const diverged = !Number.isFinite(trainLoss) || !Number.isFinite(validationLoss) || trainLoss > 1e6 || validationLoss > 1e6;

  if (diverged) {
    state.weights = previousWeights;
    state.divergenceDetected = true;
    state.isTraining = false;
    onUpdate(state);
    return { trainLoss, validationLoss };
  }

  state.trainingLosses.push(trainLoss);
  state.validationLosses.push(validationLoss);
  state.epochs += 1;

  if (validationLoss < state.bestValidationLoss - 1e-8) {
    state.bestValidationLoss = validationLoss;
    state.epochsWithoutImprovement = 0;
  } else {
    state.epochsWithoutImprovement += 1;
  }

  if (!state.overfittingWarning && hasOverfittingPattern(state.trainingLosses, state.validationLosses)) {
    state.overfittingWarning = true;
    state.overfittingEpoch = state.epochs - 4;
  }

  if (state.epochs >= 50 && state.epochsWithoutImprovement >= 30) {
    state.earlyStopped = true;
    state.isTraining = false;
  }

  onUpdate(state);
  return { trainLoss, validationLoss };
}
