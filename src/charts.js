import * as d3 from 'd3';
import { X_MIN, X_MAX } from './data.js';
import { predict } from './model.js';

const COLORS = { train: '#2f7774', validation: '#df6b4d', grid: '#e6ebe4', axis: '#78908b' };

function baseSvg(width, height, label) {
  return d3.create('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('role', 'img')
    .attr('aria-label', label);
}

function styleAxis(group) {
  group.select('.domain').attr('stroke', '#b7c6bf');
  group.selectAll('.tick line').attr('stroke', COLORS.grid);
  group.selectAll('.tick text').attr('fill', COLORS.axis).attr('font-size', 10).attr('font-family', 'DM Mono, monospace');
}

export function scatterChart(state) {
  const width = 760;
  const height = 410;
  const margin = { top: 16, right: 20, bottom: 40, left: 48 };
  const curve = d3.range(200).map((index) => {
    const x = X_MIN + ((X_MAX - X_MIN) * index) / 199;
    return { x, y: predict(state.weights, x) };
  });
  const allY = [...state.trainY, ...state.valY, ...curve.map(({ y }) => y)].filter(Number.isFinite);
  let [minY, maxY] = d3.extent(allY);
  const padding = Math.max(0.15, ((maxY ?? 1) - (minY ?? -1)) * 0.1);
  minY = (minY ?? -1) - padding;
  maxY = (maxY ?? 1) + padding;

  const xScale = d3.scaleLinear([X_MIN, X_MAX], [margin.left, width - margin.right]);
  const yScale = d3.scaleLinear([minY, maxY], [height - margin.bottom, margin.top]).nice();
  const svg = baseSvg(width, height, 'Training and validation points with the fitted polynomial curve').attr('class', 'scatter');
  const xAxis = svg.append('g').attr('transform', `translate(0,${height - margin.bottom})`).call(d3.axisBottom(xScale).ticks(5).tickSize(-(height - margin.top - margin.bottom)));
  const yAxis = svg.append('g').attr('transform', `translate(${margin.left},0)`).call(d3.axisLeft(yScale).ticks(5).tickSize(-(width - margin.left - margin.right)));
  styleAxis(xAxis); styleAxis(yAxis);

  svg.append('g').selectAll('circle').data(state.trainX).join('circle')
    .attr('cx', xScale).attr('cy', (_, index) => yScale(state.trainY[index])).attr('r', 3.1)
    .attr('fill', COLORS.train).attr('opacity', 0.6);
  svg.append('g').selectAll('circle').data(state.valX).join('circle')
    .attr('cx', xScale).attr('cy', (_, index) => yScale(state.valY[index])).attr('r', 3.4)
    .attr('fill', '#fffdf8').attr('stroke', COLORS.validation).attr('stroke-width', 1.5).attr('opacity', 0.9);
  svg.append('path').datum(curve).attr('fill', 'none').attr('stroke', COLORS.validation).attr('stroke-width', 3)
    .attr('stroke-linecap', 'round').attr('d', d3.line().x(({ x }) => xScale(x)).y(({ y }) => yScale(y)));
  svg.append('text').attr('class', 'axis-title').attr('x', width / 2).attr('y', height - 3).attr('text-anchor', 'middle').text('input x');
  svg.append('text').attr('class', 'axis-title').attr('transform', `translate(12 ${height / 2}) rotate(-90)`).attr('text-anchor', 'middle').text('output y');
  return svg.node().outerHTML;
}

export function lossChart(state) {
  const width = 760;
  const height = 190;
  const margin = { top: 16, right: 20, bottom: 40, left: 52 };
  const maxEpoch = Math.max(10, state.epochs);
  const values = [...state.trainingLosses, ...state.validationLosses].filter((value) => Number.isFinite(value) && value >= 0);
  const maxLoss = Math.max(0.1, d3.max(values) ?? 0.1);
  const xScale = d3.scaleLinear([0, maxEpoch], [margin.left, width - margin.right]);
  const yScale = d3.scaleSymlog().constant(0.01).domain([0, maxLoss * 1.08]).range([height - margin.bottom, margin.top]).nice();
  const svg = baseSvg(width, height, 'Training and validation loss over epochs').attr('class', 'loss');
  const xAxis = svg.append('g').attr('transform', `translate(0,${height - margin.bottom})`).call(d3.axisBottom(xScale).ticks(5).tickSize(-(height - margin.top - margin.bottom)));
  const yAxis = svg.append('g').attr('transform', `translate(${margin.left},0)`).call(d3.axisLeft(yScale).ticks(4, '~g').tickSize(-(width - margin.left - margin.right)));
  styleAxis(xAxis); styleAxis(yAxis);

  if (state.overfittingEpoch !== null) {
    svg.append('rect').attr('x', xScale(state.overfittingEpoch)).attr('y', margin.top)
      .attr('width', Math.max(0, width - margin.right - xScale(state.overfittingEpoch)))
      .attr('height', height - margin.top - margin.bottom).attr('fill', COLORS.validation).attr('opacity', 0.07);
  }

  const line = d3.line().x((_, index) => xScale(index + 1)).y((value) => yScale(value));
  svg.append('path').datum(state.trainingLosses).attr('fill', 'none').attr('stroke', COLORS.train).attr('stroke-width', 2.5).attr('d', line);
  svg.append('path').datum(state.validationLosses).attr('fill', 'none').attr('stroke', COLORS.validation).attr('stroke-width', 2.5).attr('d', line);
  if (state.epochs) svg.append('line').attr('x1', xScale(state.epochs)).attr('x2', xScale(state.epochs)).attr('y1', margin.top).attr('y2', height - margin.bottom).attr('stroke', '#183b3f').attr('stroke-dasharray', '3 4').attr('opacity', 0.35);
  svg.append('text').attr('class', 'axis-title').attr('x', width / 2).attr('y', height - 3).attr('text-anchor', 'middle').text('epochs');
  svg.append('text').attr('class', 'axis-title').attr('transform', `translate(12 ${height / 2}) rotate(-90)`).attr('text-anchor', 'middle').text('loss');
  return svg.node().outerHTML;
}
