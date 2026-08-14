# Poly Fit: Overfitting in Action

An interactive learning lab for polynomial regression, mini-batch stochastic gradient descent, train/validation splits, and L2 regularization. It generates a noisy multi-wave signal, trains a polynomial in the browser, and plots the fitted curve and both losses live.

## Run locally

```bash
npm install
npm run dev
```

Then open the local URL printed by Vite.

## Verify

```bash
npm test
npm run build
```

## What to try

- Degree `1`: clear underfitting.
- Degree `5`, noise `0.10`: a more balanced fit.
- Degree `15`, noise `0.50`, λ `0`: a flexible model prone to overfitting.
- Raise λ to `0.50`: coefficients shrink and the curve becomes smoother.
- Learning rate `1.0`: training should stop safely if it diverges.

The app uses an 80/20 randomized train/validation split, scales x to `[-1, 1]`, and does not regularize the bias coefficient.

## GitHub Pages

`vite.config.js` sets the base path to `/ml-regression/`. Build output is written to `dist/` and can be deployed with a GitHub Pages workflow or a `gh-pages` branch.

Further reading: [scikit-learn supervised learning](https://scikit-learn.org/stable/supervised_learning.html).
