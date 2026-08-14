import test from 'node:test';
import assert from 'node:assert/strict';
import { generateDataset, signal, splitData, X_MAX, X_MIN } from '../src/data.js';

test('generateDataset spans the full x range and is noiseless at zero noise', () => {
  const dataset = generateDataset(5, 0, () => 0.5);
  assert.equal(dataset.x[0], X_MIN);
  assert.equal(dataset.x.at(-1), X_MAX);
  assert.deepEqual(dataset.y, dataset.x.map(signal));
});

test('splitData produces disjoint 80/20 partitions', () => {
  const x = Array.from({ length: 10 }, (_, index) => index);
  const y = x.map((value) => value * 2);
  const split = splitData(x, y, 0.8, () => 0.5);
  assert.equal(split.trainX.length, 8);
  assert.equal(split.valX.length, 2);
  assert.equal(new Set([...split.trainX, ...split.valX]).size, 10);
});
