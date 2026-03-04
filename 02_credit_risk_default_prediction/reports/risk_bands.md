
# Credit Risk Bands (Policy-Ready)

A typical lending workflow consumes model scores as **probability bands**.

## Suggested Bands (example)
- **Low risk**: p(default) < 0.10 → auto-approve
- **Medium risk**: 0.10 ≤ p(default) < 0.25 → manual review / pricing adjustment
- **High risk**: p(default) ≥ 0.25 → decline or require collateral

## Monitoring
Track monthly:
- Default rate by band
- Approval rate by band
- Profit / loss by band (if available)

Update thresholds based on risk appetite and operational review capacity.
