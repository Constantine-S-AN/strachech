Output docs/assets/demo.gif

Set Theme "Catppuccin Frappe"
Set FontSize 18
Set Width 1200
Set Height 720
Set Padding 20
Set TypingSpeed 70ms

Type "python scripts/build_site.py --root ."
Enter
Sleep 2s

Type "echo \"Report HTML: $(pwd)/site/report/index.html\""
Enter
Sleep 1s

Type "echo \"Dashboard command: python -m stratcheck dashboard --results-jsonl reports/results.jsonl --db reports/paper_trading.sqlite --output reports/dashboard.html --reports-dir reports\""
Enter
Sleep 1s

Type "echo \"Site entry: $(pwd)/site/index.html\""
Enter
Sleep 2s
