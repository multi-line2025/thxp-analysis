# THxP Analysis CLI

GitHub Actionsと組み合わせて、CSVをpushするだけで解析・レポート生成が可能です。

## ローカル実行

```bash
pip install -r requirements.txt
python analyze_thxp.py --input "data/*.csv" --half-life-min 30 --freq 5T --top 3 --outdir outputs
```

## 出力
- summary_tickers.csv
- ticker_integral_derivative_summary.csv
- analysis_ticker_<ticker>.csv
- plots/*.png
