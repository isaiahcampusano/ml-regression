import { scatterChart } from '../charts.js';

export function renderScatterPlot(state) {
  return `<section class="chart-card card">
    <div class="chart-head"><span class="chart-title">Signal + fitted curve</span><div class="legend"><span style="--dot:#2f7774">train</span><span class="hollow" style="--dot:#df6b4d">validation</span><span class="line-key" style="--dot:#df6b4d">model</span></div></div>
    ${scatterChart(state)}
  </section>`;
}
