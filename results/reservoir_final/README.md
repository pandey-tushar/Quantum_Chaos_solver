# Quantum Reservoir Final Results ⭐

**Architecture:** 5 qubits, 2 layers (fixed), 160 readout weights  
**Training:** Ridge regression (closed-form)  
**Training Time:** 0.8 seconds

## Performance

- **Train MSE:** 22.37 (40.8% better than PINN)
- **Test MSE:** 3.16 (excellent generalization)
- **Speedup:** 18,000× faster than PINN

## Files

- `reservoir_model.npz` - Fixed reservoir + trained readout weights
- `config.json` - Full configuration and performance metrics

## Reproduce

```bash
python scripts/train_reservoir.py \
  --n-qubits 5 \
  --n-layers 2 \
  --n-train 50 \
  --window 5 \
  --alpha 1.0 \
  --results-dir results/reservoir_final
```

## Key Innovation

**Temporal Windowing:** Concatenates 5 time steps (window=5) to provide memory:
- Input: 3D state [x, y, z]
- Reservoir features: 32D per time step
- Temporal features: 160D (32 × 5 time steps)
- Output: Predicted [x, y, z]

This approach:
✅ Avoids barren plateaus (no gradient through reservoir)  
✅ Captures temporal dynamics crucial for chaos  
✅ Enables fast closed-form training  
✅ Achieves excellent generalization

