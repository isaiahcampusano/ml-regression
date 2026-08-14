export const X_MIN = -3;
export const X_MAX = 3;

export function gaussianRandom(random = Math.random) {
  const u = Math.max(Number.EPSILON, random());
  const v = Math.max(Number.EPSILON, random());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function signal(x) {
  return 0.5 * Math.sin(1.5 * x) + 0.3 * Math.sin(3 * x + 1) + 0.2 * Math.sin(5.5 * x - 0.5);
}

export function generateDataset(numPoints, noiseStd, random = Math.random) {
  const count = Math.max(2, Math.floor(numPoints));
  const noise = Math.max(0, Number(noiseStd));
  const x = Array.from({ length: count }, (_, index) => X_MIN + ((X_MAX - X_MIN) * index) / (count - 1));
  const y = x.map((value) => signal(value) + gaussianRandom(random) * noise);
  return { x, y };
}

function shuffle(values, random) {
  for (let index = values.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [values[index], values[swapIndex]] = [values[swapIndex], values[index]];
  }
  return values;
}

export function splitData(x, y, trainRatio = 0.8, random = Math.random) {
  if (x.length !== y.length) throw new Error('x and y must contain the same number of values.');
  if (x.length < 2) throw new Error('At least two data points are required.');
  const indices = shuffle(x.map((_, index) => index), random);
  const cutoff = Math.max(1, Math.min(indices.length - 1, Math.floor(indices.length * trainRatio)));
  const project = (list, source) => list.map((index) => source[index]);
  const train = indices.slice(0, cutoff);
  const validation = indices.slice(cutoff);
  return {
    trainX: project(train, x), trainY: project(train, y),
    valX: project(validation, x), valY: project(validation, y),
  };
}
