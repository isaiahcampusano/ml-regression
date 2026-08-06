# ml-regression
basic linear regression w larger data sets (stochastic gradient descent) -- account for overfitting and shi

I’ll craft a comprehensive handoff for an interactive web tool that teaches polynomial regression, stochastic gradient descent, and overfitting with a rich synthetic signal, clear loss curves, and a focused Learn tab. The app will use Vite + vanilla JS, be modular, and deploy to GitHub Pages.

**Key design decisions based on your answers:**
- **Dataset:** synthetic “waveform” composed of multiple sine waves plus Gaussian noise, simulating a complex real-world signal (like an audio snippet). This makes overfitting visible: a low-degree polynomial underfits, a medium degree fits the underlying waves, and a high degree fits the noise.
- **Model:** polynomial regression trained via mini‑batch SGD, with optional L2 regularization (controlled by a slider) to combat overfitting.
- **Train/validation split:** 80/20 random split of the data, both losses plotted together to illustrate when the validation loss starts rising (overfitting).
- **Controls:** minimal but powerful – just polynomial degree, learning rate, and regularization strength (λ), plus noise level and data size. A “Show Weights” panel highlights the largest coefficients and explains their meaning.
- **UI:** two tabs – “Playground” (main interactive view) and “Learn” (compact, reference-style explanations). The playground shows the data scatter + fitted curve, a live loss‑over‑epoch chart, and a weight table.
- **Edge cases:** full handling with visual warnings (e.g., “Training diverged!”, “Validation loss > 10× training loss”).

---

## Full Handoff Document for Codex / AI Coder

---

### 1. Project Overview

**Repo name suggestion:** `poly-sgd-overfitting`  
**Goal:** Build a single‑page interactive educational app that lets users explore how polynomial regression with stochastic gradient descent behaves on a complex, noisy synthetic signal. The app must clearly demonstrate underfitting, good fit, and overfitting, and explain the core concepts (SGD, train/validation split, regularization) in a dedicated “Learn” tab.

**Target user:** Someone new to machine learning who wants to see overfitting in action without needing to code.

**Deployment:** Static site via GitHub Pages.

---

### 2. Folder Structure

```
poly-sgd-overfitting/
├── index.html
├── package.json
├── vite.config.js
├── style.css                 (optional, or CSS in components)
├── src/
│   ├── main.js               (entry point, mounts app)
│   ├── state.js               (global reactive state)
│   ├── data.js                (synthetic data generation)
│   ├── model.js               (polynomial SGD logic)
│   ├── charts.js              (D3 or Canvas plotting functions)
│   ├── components/
│   │   ├── TabContainer.js
│   │   ├── PlaygroundTab.js
│   │   ├── LearnTab.js
│   │   ├── ControlPanel.js
│   │   ├── ScatterPlot.js
│   │   ├── LossPlot.js
│   │   ├── WeightDisplay.js
│   │   └── AlertBanner.js
│   └── utils.js               (math helpers)
└── README.md
```

**All files are ES modules.** Vite will handle imports. No framework; vanilla JS with direct DOM manipulation and possibly D3 for plots (see Dependencies).

---

### 3. Dependencies

- **Vite** for dev server and build.
- **D3.js v7** (or just the modules you need: `d3-scale`, `d3-axis`, `d3-line`, `d3-selection`). Use D3 to render the scatter plot + curve and the loss plot. This gives smooth, declarative SVG rendering.
- **No other libraries** (no React). Keep it lightweight.

`package.json` (after `npm init` and installs):
```json
{
  "name": "poly-sgd-overfitting",
  "version": "1.0.0",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "devDependencies": {
    "vite": "^5.0.0"
  },
  "dependencies": {
    "d3": "^7.8.0"
  }
}
```

---

### 4. Core Concepts (for the coder)

You must implement these mathematical pieces clearly. The code will be modular, so the math is inside `model.js` and `data.js`.

#### 4.1 Polynomial Regression
We model the relationship between input \(x\) (scalar, e.g., time) and output \(y\) as:
\[
y = w_0 + w_1 x + w_2 x^2 + \dots + w_d x^d
\]
where \(d\) is the polynomial degree (user-set).  
The vector of weights \(\mathbf{w} = [w_0, w_1, \dots, w_d]\).

Given a data point \((x_i, y_i)\), the prediction is:
\[
\hat{y}_i = \sum_{k=0}^{d} w_k \cdot x_i^k
\]
**Feature transformation:** for each \(x_i\), compute the vector \([1, x_i, x_i^2, \dots, x_i^d]\).

#### 4.2 Loss Function (Mean Squared Error + L2 Regularization)
For a batch of \(m\) samples:
\[
L(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2 + \frac{\lambda}{2} \sum_{k=1}^{d} w_k^2
\]
- We do NOT regularize \(w_0\) (bias term). So the sum over \(k\) starts at 1.
- \(\lambda\) (lambda) is the regularization strength, set by slider.

#### 4.3 Stochastic Gradient Descent (Mini‑Batch)
1. Shuffle the training dataset.
2. For each epoch, iterate over mini‑batches of size `batchSize` (fixed, e.g., 32).
3. For each mini‑batch, compute gradient of loss w.r.t. weights:
   - Without regularization: \(\frac{\partial L}{\partial w_k} = \frac{2}{m} \sum_{i} (\hat{y}_i - y_i) \cdot x_i^k\)
   - With L2: add \(\lambda w_k\) for \(k \ge 1\).
4. Update weights: \(w_k \leftarrow w_k - \eta \cdot \frac{\partial L}{\partial w_k}\), where \(\eta\) is learning rate.

**Implementation detail:** For numerical stability, scale the input \(x\) to the range [-1, 1] or similar. The data generation will produce \(x\) values uniformly spaced in \([-3, 3]\). Scaling before computing polynomial features prevents huge numbers when degree is high. I recommend standardizing \(x\) to zero mean and unit variance (or simply mapping [-3,3] to [-1,1]) as a preprocessing step. The `model.js` module must accept raw x, transform it, then compute features.

#### 4.4 Train/Validation Split
- Generated dataset: N points (e.g., 300).
- Randomly shuffle indices, take first 80% for training, remaining 20% for validation.
- After each epoch, compute training loss on the entire training set (or on a held‑out subset to save time) and validation loss on the validation set. (We compute both losses using the full respective sets after each epoch for accurate curves.)
- Overfitting is indicated when validation loss stops decreasing and starts to increase while training loss keeps falling.

---

### 5. Data Generation (`data.js`)

**Ground truth function:** Simulate a complex waveform (like a sports commentator’s “touchdown” shout, approximated by a mix of sine waves):
\[
f(x) = 0.5 \sin(1.5 x) + 0.3 \sin(3.0 x + 1) + 0.2 \sin(5.5 x - 0.5) + \text{noise}
\]
- \(x\) range: [-3, 3] (roughly 6 time units). You’ll generate `numPoints` evenly spaced in this interval.
- Add independent Gaussian noise with standard deviation `noiseStd` (slider, default 0.3).
- The resulting dataset is \((x_i, y_i)\).

**Export functions:**
- `generateDataset(numPoints, noiseStd)` → `{ x: number[], y: number[] }`
- `splitData(x, y, trainRatio=0.8)` → `{ trainX, trainY, valX, valY }` (shuffled)

**Important:** The dataset should be reproducible for a given seed (optional). Use `Math.random` but allow a “Generate New Data” button that calls `generateDataset` again with the current parameters.

---

### 6. State Management (`state.js`)

Central state object (reactive via a simple pub‑sub or just polled by rendering functions). Because this is vanilla JS, use a plain object and call an `updateUI()` function whenever state changes. For simplicity, use a global `appState` and a `setState(partial)` that merges and triggers re‑render.

```js
export const appState = {
  // Dataset params
  numPoints: 300,
  noiseStd: 0.3,
  xRange: [-3, 3],
  
  // Dataset (generated)
  rawX: [],
  rawY: [],
  trainX: [], trainY: [],
  valX: [], valY: [],

  // Model params
  degree: 3,                // polynomial degree, slider range 0..20
  learningRate: 0.01,       // slider range 0.0001..1, log scale maybe
  lambda: 0.0,              // regularization strength, 0..1
  batchSize: 32,            // fixed

  // Model state
  weights: [],              // length degree+1, initialized to small random
  epochs: 0,                // current epoch count
  maxEpochs: 500,           // stop after this many
  trainingLosses: [],       // arrays of length epochs, training loss per epoch
  validationLosses: [],

  // UI flags
  isTraining: false,
  trainingComplete: false,
  divergenceDetected: false,
  overfittingWarning: false,
};
```

When training starts, the loop runs asynchronously (using `setTimeout`/`requestAnimationFrame`) so the UI updates. The training loop will be inside `model.js` but managed by a function that updates state and calls an `onUpdate` callback.

**`setState` implementation:** simple, e.g., `Object.assign(appState, partial); updateUI();`.

---

### 7. Algorithm Implementation (`model.js`)

**Exports:**
- `initializeWeights(degree)` → `Float64Array` of length `degree+1`, random uniform in [-0.5, 0.5].
- `polynomialFeatures(x, degree, scaleX=true)` → array of length `degree+1` where `scaleX` normalizes `x` to [-1,1] using `xScaled = x / xRangeMax` (since xRange is [-3,3], scaling factor `xScale = 3`). Return `[1, xScaled, xScaled^2, ..., xScaled^degree]`. This scaling is crucial for high degrees.
- `predict(weights, x)` → dot product of weights and feature vector.
- `computeLoss(xArr, yArr, weights, lambda)` → MSE + L2 penalty.
- `sgdStep(batchX, batchY, weights, learningRate, lambda)` → updates weights in place using gradient descent on that batch.
- `trainEpoch(state, onUpdate)` → performs one full epoch (shuffle training data, iterate mini-batches, then compute losses) and updates state fields; calls `onUpdate()` after each epoch so UI can redraw.

**SGD step pseudocode:**
```
function sgdStep(batchX, batchY, weights, lr, lambda):
    m = batchX.length
    grad = array of zeros same as weights
    for i from 0 to m-1:
        x = batchX[i]
        y_true = batchY[i]
        feat = polynomialFeatures(x, weights.length-1)
        y_pred = dot(weights, feat)
        error = y_pred - y_true
        for k from 0 to degree:
            grad[k] += (2/m) * error * feat[k]
    // add L2 gradient
    for k from 1 to degree:
        grad[k] += lambda * weights[k]
    // update
    for k from 0 to degree:
        weights[k] -= lr * grad[k]
```

**Training loop logic (inside `trainEpoch`):**
- Check `divergenceDetected`: if any loss becomes NaN or exceeds a huge threshold (e.g., 1e6), set `divergenceDetected = true`, stop training, show alert.
- Before epoch 1, initialize weights and clear loss arrays.
- After epoch: compute training loss on full training set, validation loss on validation set. Push to arrays. Check overfitting: after some minimum epochs (say 20), if validation loss increases for several consecutive epochs (e.g., 5) while training loss decreases, set `overfittingWarning = true`. (Use a simple counter.)
- Increment `epochs`.
- If `epochs >= maxEpochs`, stop training and set `trainingComplete = true`.

---

### 8. UI Layout and Components

**Overall layout (two tabs):** The `index.html` will have a minimal shell, and `main.js` creates a tabbed container. Tabs: “Playground” and “Learn”.

- **Playground Tab** occupies the full viewport below a thin header. It is divided into:
  - **Left column (70% width):** SVG `ScatterPlot` with data points (scatter plot) and the polynomial curve (line).
  - **Right column (30% width):** `ControlPanel` with sliders, buttons, and the `WeightDisplay`.
  - **Bottom strip (below plot, full width):** `LossPlot` showing training and validation loss curves over epochs.
- **Learn Tab** occupies the full content area and contains static educational content in clean typography.

**Header:** Minimal, showing the app title “Poly Fit: Overfitting in Action” and a GitHub link.

**Responsive?** The layout should work on desktop (1024px+). On smaller screens, stack vertically.

#### Component Details

**A. `ControlPanel.js`**
- Sliders:
  - Polynomial degree (integer, 0 to 20, default 3). Use `type="range"` with step 1, display value.
  - Learning rate (float, 0.0001 to 1, log scale). Use a slider with logarithmic mapping: `value = Math.pow(10, sliderValue)` where slider goes from -4 to 0 (step 0.01). Display the actual value.
  - Regularization λ (0 to 1, step 0.01). Show current.
  - Noise std (0 to 1, step 0.01). Default 0.3.
  - Number of points (50 to 2000, step 10). Default 300.
- Buttons:
  - “Generate New Dataset” – calls data generation with current noise/points, re-splits, resets model.
  - “Reset & Retrain” – re-initializes weights and starts training from epoch 0. (Disables if already training.)
  - “Start / Pause” toggle for training.
- Model epoch counter displayed.
- Overfitting warning banner if triggered.

**B. `WeightDisplay.js`**
- Shows a table of weights: index `k` and value `w_k`. The largest absolute weights (e.g., top 3) are highlighted (maybe red if overfitting). Below the table, a dynamic message: e.g., “Small, balanced weights indicate a good fit.” or “Large weights (|w| > 5) often signal overfitting, especially with high degree and low λ.”

**C. `ScatterPlot.js`**
- SVG element with D3.
- Axes: x-axis fixed from -3 to 3; y-axis auto-scaled based on data and curve (with padding). Update axis domain when data changes.
- Scatter points: small circles for training points (blue), slightly different shape/color for validation points (orange with hollow circles, or gray). Use opacity.
- Regression curve: a line path computed by evaluating the model at 200 evenly spaced x values across the range, using current weights. Drawn on top of scatter.
- Update curve smoothly during training (redraw on each epoch tick).

**D. `LossPlot.js`**
- SVG below the main plot, height ~150px.
- X-axis: epochs, Y-axis: loss (log scale optional?). Use linear scale but if losses vary greatly, consider log scale checkbox.
- Two line paths: training loss (blue) and validation loss (red). Add legend.
- Vertical line marker at current epoch.
- Annotations: when overfitting detected, maybe shade the region where validation loss diverges.

**E. `AlertBanner.js`**
- A fixed position banner at top (or inline in control panel) that shows:
  - “⚠️ Training diverged! Try lowering the learning rate.”
  - “⚠️ Overfitting likely: validation loss is increasing while training loss decreases. Increase regularization or reduce degree.”
  - “✅ Training complete.”

---

### 9. Interaction and Animation

- When “Start” is pressed, `isTraining = true`. The training loop runs asynchronously using `setTimeout(fn, 0)` or `requestAnimationFrame` to avoid blocking. Each epoch we call `updateUI()`.
- Optionally, let users step through epochs one by one (a “Step” button) for deeper inspection, but the main play/pause is enough.
- **Speed:** by default, the training should be fast enough to see the curve evolve (e.g., 20ms delay per epoch). No extra speed slider needed.
- During training, disable the “Generate New Dataset” and degree/learning rate sliders (or allow them and retrain).
- After training stops (max epochs or pause), the final state is shown.

---

### 10. Learn Tab Content

Structure the Learn tab with concise sections, using good typography (no fancy interactions). Content to include:

**What is Polynomial Regression?**  
We try to find a curve \(y = w_0 + w_1 x + w_2 x^2 + \dots + w_d x^d\) that fits data points. The degree \(d\) controls complexity.

**What is Stochastic Gradient Descent (SGD)?**  
An algorithm that iteratively adjusts the weights \(w\) to minimize the error. Instead of using all data at once, it uses small random batches to compute gradients – faster and works with large datasets.

**Loss Function (Mean Squared Error)**  
\[ \text{MSE} = \frac{1}{m} \sum (\hat{y} - y)^2 \]  
We add L2 regularization: \( +\frac{\lambda}{2} \sum_{k=1}^d w_k^2 \) to penalize large weights, reducing overfitting.

**Train/Validation Split**  
We split data: 80% for training (to update weights), 20% for validation (to check how well the model generalizes). If the validation error starts increasing while training error drops, the model is overfitting – it’s memorizing noise.

**Overfitting & Underfitting**  
- Underfitting (degree too low): the curve is too simple and cannot capture the patterns. Both training and validation losses are high.  
- Good fit: curve captures the main trends, both losses are low and similar.  
- Overfitting (degree too high, or λ too small): the curve wiggles to pass through every training point, including noise. Training loss becomes very low, but validation loss rises. The weights often become very large.

**Visual Guide**  
Show static images or a diagram (just text description) of the three regimes. (Since we have limited space, maybe use the Playground to demonstrate, and the Learn tab explains.)

---

### 11. Edge Cases & Robustness (`model.js` and `state.js` must handle)

**a) Learning rate too high:**
- Gradient updates cause weights to explode → NaN loss or loss > 1e6.
- Detection: after each epoch, check `isNaN(loss)` or `loss > 1e6`. Set `divergenceDetected = true`, stop training, show alert “Training diverged. Reduce learning rate.” The UI shows the last good curve (before explosion) if possible; otherwise reset.

**b) Degree = 0 (constant fit):**
- Works fine, just `weights[0]` equals mean of y. The curve is a horizontal line.

**c) Very high degree (e.g., 20) with many points:**
- Polynomial features may still overflow because of scaling. Since x scaled to [-1,1], x^k stays within [-1,1], so no overflow. However, the model can overfit wildly. This is desired; the UI must show large weights and overfitting warning.

**d) Batch size larger than dataset:**
- In `sgdStep`, if `batchSize` > number of training points, we use full batch gradient descent. Code should handle: if `batchSize >= trainX.length`, set `batchSize = trainX.length` for that loop.

**e) All validation points identical or dataset too small:**
- Not an issue with synthetic data, but if `numPoints` is very low (e.g., 10), the split might leave only 2 validation points. Loss calculations still work. No crash.

**f) Regularization lambda very high:**
- Weights will be driven toward zero. The fit becomes a nearly horizontal line. That’s fine; it demonstrates underfitting due to excessive regularization.

**g) No data generated yet:**
- On load, automatically generate a dataset with default parameters and start training so the app is immediately interactive. Or provide a “Generate & Train” button.

**h) User changes parameters during training:**
- Sliders for degree, learning rate, lambda, noise, numPoints should be disabled while `isTraining` is true, to prevent inconsistency. After training stops, re‑enable. (Alternatively, allow change and auto‑reset.) We'll disable.

**i) Training takes too many epochs:**
- `maxEpochs` = 500 is enough. If loss stagnates, we might add early stopping: if validation loss doesn't improve for 30 epochs, stop and show “Early stopping.” Optional, but helpful.

**j) Visualization performance:**
- Redrawing whole plot every epoch with D3 is fine for 500 epochs. Use `selection.join` to update scatter points efficiently, but we can just clear and redraw.

**k) Accessibility:**
- Basic: labels for sliders, focus outlines, maybe ARIA for alerts.

---

### 12. Testing Checklist (for after build)

- [ ] Open page: dataset generated, training auto‑starts, curve appears and evolves.
- [ ] Adjust degree to 1: simple line, underfits. Losses both high.
- [ ] Set degree to 5, λ=0, noise=0.1: good fit. Training and validation losses decrease together.
- [ ] Set degree to 15, noise=0.5: see overfitting: wiggly curve, validation loss rises, overfitting warning appears. Weights table shows large values.
- [ ] Increase λ to 0.5 with high degree: curve becomes smoother, weights shrink.
- [ ] Set learning rate to 1.0 (too high): diverge quickly, alert shown, stop.
- [ ] Click “Generate New Dataset”: new points, model resets and retrains.
- [ ] Toggle “Pause” during training: stops, curve freezes.
- [ ] Resize window: layout remains usable.
- [ ] Switch to Learn tab: content readable, no broken layout.
- [ ] All sliders respond and update display values correctly.
- [ ] No console errors.
- [ ] Deploy to GitHub Pages and test live.

---

### 13. Deployment to GitHub Pages

1. Set `base: '/poly-sgd-overfitting/'` in `vite.config.js` (or adjust according to your repo name).
2. Run `npm run build`. Output in `dist/`.
3. Use `gh-pages` branch or GitHub Actions to deploy. Provide a simple GitHub Actions workflow (optional) or manual instructions in README.
4. Ensure `index.html` and assets are served correctly.

---

### 14. Additional Notes for the Coder

- **State updates:** Because we don't use a framework, every component that needs to update can subscribe to a global `updateUI` event. I suggest a simple event emitter or just calling a `renderAll()` function from `main.js` after `setState`. The loss plot and scatter plot will re‑render from `appState` arrays.
- **Optimization:** To keep the UI responsive during training, train one epoch per animation frame or per tick. Use a recursive `setTimeout(fn, 0)` loop that stops when `isTraining` becomes false.
- **Scaling:** Use the scaling factor `xScale = 3` because x range is [-3,3]. Inside `polynomialFeatures`, map `xScaled = x / 3`.
- **Weights display:** Show absolute values with color coding: green if |w| < 2, yellow if 2‑5, red if >5 (thresholds adjustable).
- **Learn tab styling:** Use a clean serif font for formulas (maybe CSS `math-style`). Keep it brief; provide a link to further reading.

---

This handoff is designed to be executed file‑by‑file by an AI coder. The structure, algorithms, and UI components are fully specified. Now, go build and deploy!
