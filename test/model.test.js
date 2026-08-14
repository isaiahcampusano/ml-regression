import test from 'node:test';
import assert from 'node:assert/strict';
import { computeLoss, polynomialFeatures, predict, sgdStep } from '../src/model.js';

test('polynomialFeatures scales x to a stable range', () => {
  assert.deepEqual([...polynomialFeatures(3, 3)], [1, 1, 1, 1]);
  assert.deepEqual([...polynomialFeatures(-3, 3)], [1, -1, 1, -1]);
});

test('predict evaluates polynomial weights', () => {
  assert.equal(predict(Float64Array.from([1, 2, 3]), 3), 6);
});

test('L2 loss excludes the bias term', () => {
  const xs = [0];
  const ys = [1];
  assert.equal(computeLoss(xs, ys, Float64Array.from([1, 2]), 0.5), 1);
});

test('an SGD step reduces loss on a simple constant target', () => {
  const xs = [-3, -1, 1, 3];
  const ys = [2, 2, 2, 2];
  const weights = Float64Array.from([0]);
  const before = computeLoss(xs, ys, weights);
  sgdStep(xs, ys, weights, 0.1);
  assert.ok(computeLoss(xs, ys, weights) < before);
});
