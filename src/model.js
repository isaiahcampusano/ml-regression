import { X_MAX } from './data.js';

export const features = (x, degree) => {
  const scaled = x / X_MAX;
  const values = [1];
  for (let k = 1; k <= degree; k += 1) values.push(values[k - 1] * scaled);
  return values;
};

export const predict = (weights, x) => features(x, weights.length - 1).reduce((sum, value, index) => sum + weights[index] * value, 0);

export function loss(xs, ys, weights, lambda = 0) {
  if (!xs.length) return 0;
  let total = 0;
  xs.forEach((x, index) => {
    const error = predict(weights, x) - ys[index];
    total += error * error;
  });
  const penalty = weights.slice(1).reduce((sum, weight) => sum + weight * weight, 0) * lambda / 2;
  return total / xs.length + penalty;
}

export function initializeWeights(degree) {
  return Float64Array.from({ length: degree + 1 }, () => (Math.random() - 0.5) * 0.1);
}

export function sgdStep(xs, ys, weights, learningRate, lambda) {
  const gradient = new Float64Array(weights.length);
  xs.forEach((x, index) => {
    const row = features(x, weights.length - 1);
    const error = predict(weights, x) - ys[index];
    row.forEach((value, degree) => { gradient[degree] += (2 / xs.length) * error * value; });
  });
  for (let degree = 1; degree < weights.length; degree += 1) gradient[degree] += lambda * weights[degree];
  weights.forEach((_, index) => { weights[index] -= learningRate * gradient[index]; });
}

export function trainEpoch(state) {
  const order = state.trainX.map((_, index) => index).sort(() => Math.random() - 0.5);
  const batchSize = Math.min(state.batchSize, state.trainX.length);
  for (let start = 0; start < order.length; start += batchSize) {
    const batch = order.slice(start, start + batchSize);
    sgdStep(batch.map((index) => state.trainX[index]), batch.map((index) => state.trainY[index]), state.weights, state.learningRate, state.lambda);
  }
  const trainLoss = loss(state.trainX, state.trainY, state.weights, state.lambda);
  const validationLoss = loss(state.valX, state.valY, state.weights, state.lambda);
  state.trainingLosses.push(trainLoss);
  state.validationLosses.push(validationLoss);
  state.epochs += 1;
  const recent = state.validationLosses.slice(-5);
  state.overfittingWarning = state.epochs > 20 && recent.length === 5 && recent.every((value, index) => index === 0 || value > recent[index - 1]) && trainLoss < state.trainingLosses[Math.max(0, state.trainingLosses.length - 6)];
  state.divergenceDetected = !Number.isFinite(trainLoss) || !Number.isFinite(validationLoss) || trainLoss > 1e6 || validationLoss > 1e6;
  return { trainLoss, validationLoss };
}
