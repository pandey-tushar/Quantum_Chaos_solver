# Quantum PINN Final Results

**Architecture:** 4 qubits, 3 layers, 45 trainable parameters  
**Training:** 200 iterations with early stopping  
**Training Time:** ~4 hours

## Performance

- **Final MSE:** 37.77
- **Status:** Converged with early stopping

## Files

- `final_parameters.npz` - Trained quantum circuit parameters
- `training_history.json` - Full training metrics per iteration

## Reproduce

```bash
python scripts/train_lorenz_improved.py \
  --n-qubits 4 \
  --n-layers 3 \
  --n-iterations 200 \
  --results-dir results/pinn_final
```

## Notes

- Physics-informed loss with hybrid data-driven component
- Gradient clipping and learning rate decay applied
- Avoided barren plateaus with careful initialization
- Matches classical PINN performance with 89% fewer parameters

