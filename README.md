# Arbitrage Engine v1.0

A paper-trading arbitrage simulation system. All trades are simulated — no real money moves.

## Engines

| Engine | Strategy | Capital |
|--------|----------|---------|
| Engine 1 | Funding Rate Harvester | $3,000 |
| Engine 2 | Polymarket Reward Farmer | $3,000 |
| Engine 3 | Flash Loan Arbitrage | $1,500 |
| Engine 4 | Triangular Arbitrage | $1,500 |
| Engine 5 | Cross-Exchange Spread | $1,000 |

## Setup

```bash
pip install -r requirements.txt
python runner.py
```

## Dashboard

The web dashboard is served at `http://localhost:5000` (or your Render URL).

## Deployment

Deploy to Render using the included `render.yaml` blueprint.
