
# Campaign ROI Simulation (Template + Filled Example)

This project includes a lift curve by score decile. To convert that into business decisions, estimate ROI.

## Inputs
- Contacts available: 100,000
- Cost per contact: $0.65
- Margin per conversion: $120

## Strategy
Target the **top deciles** until budget or call-center capacity is reached.

## Example (Filled)
If top decile response rate is 18% and baseline is 6%:
- Expected conversions (top decile 10,000 contacts): 1,800
- Cost: 10,000 × $0.65 = $6,500
- Margin: 1,800 × $120 = $216,000
- Estimated ROI: (216,000 − 6,500) / 6,500 ≈ **32.2×**

Use `reports/lift_by_decile.csv` to compute decile-specific ROI and pick the optimal cutoff.
