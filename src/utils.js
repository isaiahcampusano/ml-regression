export function formatNumber(value, digits = 3) {
  if (!Number.isFinite(value)) return '—';
  if (Math.abs(value) >= 1000 || (value !== 0 && Math.abs(value) < 0.001)) return value.toExponential(2);
  return value.toFixed(digits);
}
