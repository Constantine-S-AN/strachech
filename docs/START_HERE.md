# START HERE

If you only have 5 minutes, use this page as the navigation hub.

## 1) Live Demo (Pages)

- Repo-relative: [`../site/index.html`](../site/index.html)
- Pages URL template: `https://<your-user>.github.io/<your-repo>/`

## 2) 5-min Quickstart

```bash
python -m pip install -e ".[dev]"
python scripts/make_demo_assets.py --output data/QQQ.csv --periods 240 --seed 7
python -m stratcheck run --config configs/examples/buy_and_hold.toml
python -m stratcheck dashboard --results-jsonl reports/results.jsonl --db reports/paper_trading.sqlite --output reports/dashboard.html --reports-dir reports
python scripts/build_site.py --root .
```

## 3) Case Study

- [`./case-study.md`](./case-study.md)
- Site page after build: [`../site/case-study/index.html`](../site/case-study/index.html)

## 4) Interview Pack

- Interview script: [`./interview.md`](./interview.md)
- Resume bullets: [`./resume_bullets.md`](./resume_bullets.md)
