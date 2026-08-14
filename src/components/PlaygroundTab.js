import { renderAlert } from './AlertBanner.js';
import { renderControls } from './ControlPanel.js';
import { renderLossPlot } from './LossPlot.js';
import { renderScatterPlot } from './ScatterPlot.js';
import { renderWeights } from './WeightDisplay.js';

export function renderPlayground(state) {
  return `${renderAlert(state)}<div class="layout"><div class="visual-column">${renderScatterPlot(state)}${renderLossPlot(state)}</div><aside>${renderControls(state)}${renderWeights(state)}</aside></div>`;
}
