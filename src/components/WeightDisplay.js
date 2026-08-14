import { formatNumber } from '../utils.js';

export function renderWeights(state) {
  const ranked = [...state.weights].map((value, index) => ({ value, index })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const top = new Set(ranked.slice(0, 3).map(({ index }) => index));
  const largest = ranked[0]?.value ?? 0;
  const rows = [...state.weights].map((value, index) => {
    const magnitude = Math.abs(value);
    const level = magnitude > 5 ? 'hot' : magnitude >= 2 ? 'warm' : '';
    return `<tr class="${level} ${top.has(index) ? 'top-weight' : ''}"><td>w<sub>${index}</sub></td><td>${formatNumber(value)}</td></tr>`;
  }).join('');
  const note = Math.abs(largest) > 5
    ? 'Large coefficients can signal a curve that is chasing noise. Try increasing λ.'
    : 'The three largest coefficients are highlighted. Balanced weights usually track the broader signal.';
  return `<section class="weight-card card" aria-labelledby="weight-title"><h2 id="weight-title">Model weights</h2><div class="weight-scroll"><table class="weights"><thead><tr><th>term</th><th>value</th></tr></thead><tbody>${rows}</tbody></table></div><p class="weight-note">${note}</p></section>`;
}
