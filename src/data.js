export const X_MIN = -3;
export const X_MAX = 3;

function gaussian() {
  const u = 1 - Math.random();
  const v = 1 - Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function signal(x) {
  return 0.5 * Math.sin(1.5 * x) + 0.3 * Math.sin(3 * x + 1) + 0.2 * Math.sin(5.5 * x - 0.5);
}

export function generateDataset(numPoints, noiseStd) {
  const x = [];
  const y = [];
  for (let i = 0; i < numPoints; i += 1) {
    const value = X_MIN + ((X_MAX - X_MIN) * i) / (numPoints - 1);
    x.push(value);
    y.push(signal(value) + gaussian() * noiseStd);
  }
  return { x, y };
}

export function splitData(x, y, trainRatio = 0.8) {
  const indices = x.map((_, index) => index).sort(() => Math.random() - 0.5);
  const cutoff = Math.max(1, Math.min(indices.length - 1, Math.floor(indices.length * trainRatio)));
  const take = (list) => list.map((index) => x[index]);
  const takeY = (list) => list.map((index) => y[index]);
  return {
    trainX: take(indices.slice(0, cutoff)), trainY: takeY(indices.slice(0, cutoff)),
    valX: take(indices.slice(cutoff)), valY: takeY(indices.slice(cutoff))
  };
}
