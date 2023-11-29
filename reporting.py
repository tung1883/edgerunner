"""Generate HTML performance report."""
import json, os

TEMPLATE = """<!DOCTYPE html>
<html><head><title>Strategy Report</title></head>
<body>
<h1>Backtest Report</h1>
<table border="1">
<tr><th>Metric</th><th>Value</th></tr>
{rows}
</table>
</body></html>"""

def generate_report(metrics: dict, output_path="outputs/report.html"):
    rows = "\n".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.items())
    html = TEMPLATE.format(rows=rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    return output_path

def save_metrics_json(metrics: dict, path="outputs/metrics.json"):
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: float(v) if hasattr(v, '__float__') else v for k, v in metrics.items()}, f, indent=2)
