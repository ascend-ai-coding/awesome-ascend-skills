# Configuration Guide

## Environment Variables

### Email Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `SMTP_SERVER` | SMTP server address | smtp.163.com |
| `SMTP_PORT` | SMTP server port | 465 |
| `SMTP_USER` | SMTP username (email address) | (empty) |
| `SMTP_PASSWORD` | SMTP password or auth code | (empty) |
| `INFERENCEX_EMAIL_TO` | Report recipient email | your-email@example.com |
| `INFERENCEX_EMAIL_FROM` | Report sender display name | inferencex-reporter@openclaw.local |

### Example: 163 Mail Configuration

```bash
export SMTP_SERVER="smtp.163.com"
export SMTP_PORT="465"
export SMTP_USER="your-email@163.com"
export SMTP_PASSWORD="your-auth-code"  # Not login password!
export INFERENCEX_EMAIL_TO="recipient@example.com"
```

### Example: Gmail Configuration

```bash
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="465"
export SMTP_USER="your-email@gmail.com"
export SMTP_PASSWORD="your-app-password"  # App-specific password
export INFERENCEX_EMAIL_TO="recipient@example.com"
```

## Cron Job Setup

### Daily Report at 9:00 AM

```bash
# Edit crontab
crontab -e

# Add line
0 9 * * * cd /path/to/inferencex-report && python3 scripts/inferencex_api_report.py >> logs/cron.log 2>&1
```

### Weekly Report on Mondays

```bash
0 9 * * 1 cd /path/to/inferencex-report && python3 scripts/inferencex_api_report.py >> logs/cron.log 2>&1
```

## Data Directory Structure

```
data/
├── inferencex_summary_YYYY-MM-DD.csv    # Daily CSV report
├── inferencex_summary_YYYY-MM-DD.json   # Raw data for comparison
└── email_YYYY-MM-DD.html                # Email content backup
```

## Customization

### Modify Models List

Edit `scripts/inferencex_api_report.py`:

```python
MODELS = {
    'DeepSeek-R1-0528': ['dsr1'],
    'gpt-oss-120b': ['gptoss120b'],
    # Add your models here
    'Your-Model-Name': ['internal_db_key'],
}
```

### Change Performance Thresholds

Edit the `create_8k1k_tables` function to modify color coding:

```python
# Current thresholds
bg_color = "#e8f5e9" if tput > 10000 else "#fff3e0" if tput > 5000 else "white"

# Custom thresholds
bg_color = "#e8f5e9" if tput > 15000 else "#fff3e0" if tput > 8000 else "white"
```

### Change Interactivity Filter

Default filter for 8k1k table: `interactivity > 20 tps`

Edit in `process_api_data` function:

```python
if interactivity > 20 and tput > 0:  # Change 20 to your threshold
```

## Troubleshooting

### Email Not Sending

1. Check SMTP credentials are set correctly
2. Verify SMTP server allows less secure apps (for Gmail)
3. Use app-specific password instead of account password
4. Check firewall settings for SMTP port

### No Data Retrieved

1. Verify internet connection
2. Check API endpoint is accessible: `curl -I https://inferencex.semianalysis.com/api/v1/benchmarks`
3. Check if models list is up to date

### High Memory Usage

For large datasets, process data in batches:

```python
# Process one model at a time
for model_name in MODELS.keys():
    records = fetch_api_data(model_name)
    process_and_save(records)  # Save intermediate results
```
