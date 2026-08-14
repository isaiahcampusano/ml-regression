export function renderAlert(state) {
  if (state.divergenceDetected) return '<div class="alert warning" role="alert">⚠ Training diverged. The last stable curve is shown; lower the learning rate and retrain.</div>';
  if (state.overfittingWarning) return '<div class="alert warning" role="alert">⚠ Overfitting likely: validation loss rose while training loss fell. Increase λ or reduce the degree.</div>';
  if (state.earlyStopped) return '<div class="alert" role="status">✓ Early stopping paused training after validation loss stopped improving.</div>';
  if (state.trainingComplete) return '<div class="alert" role="status">✓ Training complete. Compare the loss curves to judge generalization.</div>';
  return '';
}
