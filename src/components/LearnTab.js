export function renderLearn() {
  return `<article class="learn">
    <header class="learn-intro"><p class="eyebrow">The mental model</p><h2>A curve can learn the signal — or memorize the noise.</h2><p>Use the Playground to compare these three regimes. Watch the shape of the curve, the distance between the two loss lines, and the size of the coefficients.</p></header>
    <div class="regime-grid" aria-label="Fit quality guide">
      <section class="regime under"><span>01</span><h3>Underfit</h3><p>Degree too low or λ too high. The curve misses the pattern, so both losses stay high.</p></section>
      <section class="regime good"><span>02</span><h3>Good fit</h3><p>The main waves are captured. Training and validation loss are both low and remain close.</p></section>
      <section class="regime over"><span>03</span><h3>Overfit</h3><p>Degree too high and λ too low. Training loss falls while validation loss turns upward.</p></section>
    </div>
    <div class="learn-grid">
      <section class="card"><h3>Polynomial regression</h3><p>A polynomial combines powers of x. Its degree controls how many bends the fitted curve can make.</p><div class="formula">ŷ = w₀ + w₁x + w₂x² + … + w<sub>d</sub>x<sup>d</sup></div></section>
      <section class="card"><h3>Mini-batch SGD</h3><p>Stochastic gradient descent repeatedly nudges each weight downhill. Small random batches make updates efficient on larger datasets.</p><div class="formula">w ← w − η · gradient</div></section>
      <section class="card"><h3>Loss + regularization</h3><p>Mean squared error measures prediction mistakes. L2 regularization adds a cost for large non-bias weights, encouraging a smoother fit.</p><div class="formula">loss = MSE + λ⁄2 · Σ<sub>k=1</sub><sup>d</sup> w<sub>k</sub>²</div></section>
      <section class="card"><h3>Train vs. validation</h3><p>The model learns from 80% of the points. The held-out 20% tests whether the model generalizes to data it did not train on.</p><ul><li><strong>Training loss</strong> measures the points used for updates.</li><li><strong>Validation loss</strong> measures unseen held-out points.</li></ul></section>
    </div>
    <section class="try-card card"><div><p class="eyebrow">Try this</p><h3>Make overfitting visible</h3></div><ol><li>Set degree to 15 and noise to 0.50.</li><li>Keep λ at 0 and retrain.</li><li>Then raise λ to 0.50 and compare the curve and weights.</li></ol></section>
  </article>`;
}
