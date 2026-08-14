import { lossChart } from '../charts.js';
import { formatNumber } from '../utils.js';

export function renderLossPlot(state) {
  const train = state.trainingLosses.at(-1);
  const validation = state.validationLosses.at(-1);
  return `<section class="chart-card loss-card card">
    <div class="chart-head"><span class="chart-title">Loss over time</span><div class="legend"><span style="--dot:#2f7774">training ${formatNumber(train)}</span><span style="--dot:#df6b4d">validation ${formatNumber(validation)}</span></div></div>
    ${lossChart(state)}
  </section>`;
}
