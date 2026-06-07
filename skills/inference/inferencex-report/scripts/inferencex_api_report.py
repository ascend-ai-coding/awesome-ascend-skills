#!/usr/bin/env python3
"""
InferenceX Daily Report Generator (API Version)
- Fetches benchmark data from InferenceX API
- Generates CSV performance reports
- Detects data changes (new combinations, performance variations)
- Sends email reports with 8k1k detailed performance tables
"""

import json
import csv
import os
import sys
import subprocess
import gzip
from datetime import datetime, timedelta
from collections import defaultdict
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Configuration
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = f"{WORKSPACE}/data"
EMAIL_TO = os.environ.get('INFERENCEX_EMAIL_TO', 'your-email@example.com')
EMAIL_FROM = os.environ.get('INFERENCEX_EMAIL_FROM', 'inferencex-reporter@openclaw.local')

# API Configuration
API_BASE_URL = "https://inferencex.semianalysis.com/api/v1"

# Model list (display name -> internal DB keys)
MODELS = {
    'DeepSeek-R1-0528': ['dsr1'],
    'gpt-oss-120b': ['gptoss120b'],
    'Llama-3.3-70B-Instruct-FP8': ['llama70b'],
    'Qwen-3.5-397B-A17B': ['qwen3.5'],
    'Kimi-K2.5': ['kimik2.5', 'kimik2.6'],
    'MiniMax-M2.5': ['minimaxm2.5', 'minimaxm2.7'],
    'GLM-5': ['glm5', 'glm5.1'],
    'DeepSeek-V4-Pro': ['dsv4'],
}

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)


def get_today_date():
    return datetime.now().strftime("%Y-%m-%d")


def get_yesterday_date():
    return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def fetch_api_data(model_display_name, date=None, exact=False):
    """Fetch data from API"""
    url = f"{API_BASE_URL}/benchmarks?model={model_display_name}"
    if date:
        url += f"&date={date}"
    if exact:
        url += "&exact=true"
    
    try:
        result = subprocess.run(
            ["curl", "-s", url],
            capture_output=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"  Error: curl failed - {result.stderr.decode()}")
            return []
        
        # Decompress gzip data (or use directly if not gzip)
        try:
            # Check if gzip format
            if result.stdout[:2] == b'\x1f\x8b':  # gzip magic number
                data = gzip.decompress(result.stdout)
                content = data.decode('utf-8').strip()
            else:
                # Direct JSON data
                content = result.stdout.decode('utf-8').strip()
            
            # API returns JSON Array format
            if content.startswith('['):
                records = json.loads(content)
                return records if isinstance(records, list) else []
            else:
                # Try JSON Lines format
                records = []
                for line in content.split('\n'):
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                return records
        except Exception as e:
            print(f"  Error: decompress/parse failed - {e}")
            return []
            
    except Exception as e:
        print(f"  Error: request failed - {e}")
        return []


def get_latest_date_from_api():
    """Get latest data date from API"""
    print("Querying latest data date...")
    
    # Try to get DeepSeek-R1-0528 data to find latest date
    records = fetch_api_data('DeepSeek-R1-0528')
    
    if not records:
        return None
    
    # Extract all dates and find latest
    dates = set()
    for r in records:
        if 'date' in r:
            dates.add(r['date'])
    
    if not dates:
        return None
    
    latest_date = max(dates)
    print(f"  Latest data date: {latest_date}")
    return latest_date


def fetch_all_models_data(target_date=None):
    """Fetch data for all models"""
    print(f"\nFetching data from API...")
    
    all_records = []
    
    for model_name in MODELS.keys():
        print(f"  Fetching {model_name}...")
        
        if target_date:
            # Get specific date data
            records = fetch_api_data(model_name, date=target_date, exact=True)
        else:
            # Get latest data
            records = fetch_api_data(model_name)
        
        print(f"    Retrieved {len(records)} records")
        all_records.extend(records)
    
    print(f"\nTotal records: {len(all_records)}")
    return all_records


def process_api_data(records):
    """Process API returned data"""
    # Aggregate data - group by configuration
    combo_data = defaultdict(lambda: {
        'hardware': '', 'model': '', 'framework': '', 'precision': '',
        'max_tput': 0, 'avg_tputs': [],
        'best_ttft': float('inf'), 'best_tpot': float('inf'),
        'test_count': 0, 'gpu_count': 0,
        'seq_lens': set(),
        'dates': set()
    })
    
    # Specifically store 8k1k data
    data_8k1k = defaultdict(lambda: {
        'hardware': '', 'model': '', 'framework': '', 'precision': '',
        'max_tput': 0, 'avg_tputs': [],
        'test_count': 0
    })
    
    for r in records:
        # Extract configuration info
        hardware = r.get('hardware', '')
        model = r.get('model', '')
        framework = r.get('framework', '')
        precision = r.get('precision', '')
        
        key = (hardware, model, framework, precision)
        
        d = combo_data[key]
        d['hardware'] = hardware
        d['model'] = model
        d['framework'] = framework
        d['precision'] = precision
        d['gpu_count'] = max(d['gpu_count'], r.get('num_prefill_gpu', 0) + r.get('num_decode_gpu', 0))
        
        isl = r.get('isl', 0)
        osl = r.get('osl', 0)
        seq_combo = f"{isl//1024}k{osl//1024}k"
        d['seq_lens'].add(seq_combo)
        
        date = r.get('date', '')
        if date:
            d['dates'].add(date)
        
        metrics = r.get('metrics', {})
        tput = metrics.get('tput_per_gpu', 0)
        if tput > 0:
            d['max_tput'] = max(d['max_tput'], tput)
            d['avg_tputs'].append(tput)
            d['best_ttft'] = min(d['best_ttft'], metrics.get('mean_ttft', float('inf')))
            d['best_tpot'] = min(d['best_tpot'], metrics.get('mean_tpot', float('inf')))
            d['test_count'] += 1
        
        # Specifically collect 8k1k data (ISL=8192, OSL=1024), with interactivity > 20
        if isl == 8192 and osl == 1024:
            interactivity = metrics.get('mean_intvty', 0)
            if interactivity > 20 and tput > 0:
                d8k = data_8k1k[key]
                d8k['hardware'] = hardware
                d8k['model'] = model
                d8k['framework'] = framework
                d8k['precision'] = precision
                d8k['max_tput'] = max(d8k['max_tput'], tput)
                d8k['avg_tputs'].append(tput)
                d8k['test_count'] += 1
    
    # Calculate averages
    for key in combo_data:
        avg_list = combo_data[key]['avg_tputs']
        combo_data[key]['avg_tput'] = sum(avg_list) / len(avg_list) if avg_list else 0
    
    for key in data_8k1k:
        avg_list = data_8k1k[key]['avg_tputs']
        data_8k1k[key]['avg_tput'] = sum(avg_list) / len(avg_list) if avg_list else 0
    
    # Find data date range
    all_dates = set()
    for d in combo_data.values():
        all_dates.update(d['dates'])
    
    data_date = max(all_dates) if all_dates else get_today_date()
    
    return combo_data, data_8k1k, data_date


def generate_summary_csv(combo_data, output_path):
    """Generate summary CSV"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Hardware', 'Model', 'Framework', 'Precision', 'GPU Count', 
                        'Max Throughput (tok/s/gpu)', 'Avg Throughput (tok/s/gpu)',
                        'Best TTFT (ms)', 'Best TPOT (ms)', 'Test Count', 'Sequence Length Combos'])
        
        for key, d in sorted(combo_data.items()):
            writer.writerow([
                d['hardware'].upper(),
                d['model'],
                d['framework'],
                d['precision'],
                d['gpu_count'],
                f"{d['max_tput']:.2f}",
                f"{d['avg_tput']:.2f}",
                f"{d['best_ttft']:.2f}" if d['best_ttft'] != float('inf') else 'N/A',
                f"{d['best_tpot']:.2f}" if d['best_tpot'] != float('inf') else 'N/A',
                d['test_count'],
                ','.join(sorted(d['seq_lens']))
            ])


def load_previous_data(date):
    """Load previous day's data"""
    prev_file = f"{DATA_DIR}/inferencex_summary_{date}.json"
    if os.path.exists(prev_file):
        with open(prev_file, 'r') as f:
            data = json.load(f)
            result = {}
            for key_str, d in data.items():
                key = eval(key_str) if key_str.startswith('(') else key_str
                result[key] = d
            return result
    return None


def detect_changes(current_data, previous_data):
    """Detect data changes"""
    if not previous_data:
        return "First run, no historical data for comparison", []
    
    changes = []
    current_dict = {k: v for k, v in current_data.items()}
    prev_dict = {k: v for k, v in previous_data.items()}
    
    # Detect new combinations
    new_combos = set(current_dict.keys()) - set(prev_dict.keys())
    for combo in new_combos:
        d = current_dict[combo]
        changes.append(f"NEW: {combo[0].upper()}/{combo[1]}/{combo[2]}/{combo[3]} - Max Throughput: {d['max_tput']:.2f}")
    
    # Detect performance changes
    for combo in current_dict:
        if combo in prev_dict:
            curr_max = current_dict[combo]['max_tput']
            prev_max = prev_dict[combo]['max_tput']
            if prev_max > 0:
                change_pct = ((curr_max - prev_max) / prev_max) * 100
                if abs(change_pct) > 5:
                    direction = "UP" if change_pct > 0 else "DOWN"
                    changes.append(f"{direction}: {combo[0].upper()}/{combo[1]}/{combo[2]} {change_pct:+.1f}% ({prev_max:.0f} -> {curr_max:.0f})")
    
    if not changes:
        return "No significant changes", []
    
    return f"Found {len(changes)} updates", changes


def generate_summary_stats(data):
    """Generate summary statistics"""
    stats = {
        'total_combos': len(data),
        'hardwares': set(),
        'models': set(),
        'frameworks': set(),
        'max_tput_overall': 0,
        'max_tput_info': '',
        'nvidia_best': 0,
        'amd_best': 0
    }
    
    nvidia_hw = {'B200', 'B300', 'GB200', 'GB300', 'H100', 'H200'}
    amd_hw = {'MI300X', 'MI325X', 'MI355X'}
    
    for key, d in data.items():
        stats['hardwares'].add(d['hardware'])
        stats['models'].add(d['model'])
        stats['frameworks'].add(d['framework'])
        
        if d['max_tput'] > stats['max_tput_overall']:
            stats['max_tput_overall'] = d['max_tput']
            stats['max_tput_info'] = f"{d['hardware'].upper()}/{d['model']}/{d['framework']}/{d['precision']}"
        
        hw = d['hardware'].upper()
        if hw in nvidia_hw and d['max_tput'] > stats['nvidia_best']:
            stats['nvidia_best'] = d['max_tput']
        if hw in amd_hw and d['max_tput'] > stats['amd_best']:
            stats['amd_best'] = d['max_tput']
    
    return stats


def create_8k1k_tables(data_8k1k):
    """Generate detailed tables for 8k1k sequence length"""
    
    # Group by hardware, get best performance for each hardware-model combination
    hw_model_best = defaultdict(lambda: defaultdict(lambda: {'tput': 0, 'framework': '', 'precision': ''}))
    
    for key, d in data_8k1k.items():
        hw = d['hardware'].upper()
        model = d['model']
        if d['max_tput'] > hw_model_best[hw][model]['tput']:
            hw_model_best[hw][model] = {
                'tput': d['max_tput'],
                'framework': d['framework'],
                'precision': d['precision']
            }
    
    # Get all models
    all_models = set()
    for hw_models in hw_model_best.values():
        all_models.update(hw_models.keys())
    all_models = sorted(all_models)
    
    # Hardware order: NVIDIA first, AMD second
    nvidia_order = ['B200', 'B300', 'GB200', 'GB300', 'H200', 'H100']
    amd_order = ['MI355X', 'MI325X', 'MI300X']
    hw_order = [hw for hw in nvidia_order if hw in hw_model_best] + [hw for hw in amd_order if hw in hw_model_best]
    
    # Generate table HTML
    html = """
<h3>8k1k Sequence Length Performance (ISL=8192, OSL=1024)</h3>
<p style="font-size: 12px; color: #666;">Unit: tokens/s/GPU | Shows best performance for each hardware-model combination with framework/precision</p>
<table border="1" cellpadding="5" style="border-collapse: collapse; font-size: 12px;">
<tr style="background-color: #f0f0f0;">
<th>Hardware</th>
"""
    
    # Header: models
    for model in all_models:
        html += f"<th>{model}</th>"
    html += "</tr>\n"
    
    # Data rows
    for hw in hw_order:
        html += f'<tr><td style="background-color: #f9f9f9; font-weight: bold;">{hw}</td>'
        for model in all_models:
            if model in hw_model_best[hw]:
                d = hw_model_best[hw][model]
                tput = d['tput']
                fw = d['framework']
                prec = d['precision']
                # Highlight high performance cells
                bg_color = "#e8f5e9" if tput > 10000 else "#fff3e0" if tput > 5000 else "white"
                html += f'<td style="background-color: {bg_color};">{tput:,.0f}<br/><span style="font-size: 10px; color: #666;">{fw}/{prec}</span></td>'
            else:
                html += '<td style="color: #ccc;">-</td>'
        html += "</tr>\n"
    
    html += "</table>\n"
    
    # Add legend
    html += """
<p style="font-size: 11px; color: #666;">
Legend: <span style="background-color: #e8f5e9; padding: 2px 5px;">Green</span> > 10k tok/s/GPU | 
<span style="background-color: #fff3e0; padding: 2px 5px;">Orange</span> > 5k tok/s/GPU
</p>
"""
    
    return html


def create_email_content(date, change_summary, changes, stats, data, data_8k1k):
    """Create email content"""
    
    # Part 1: Update status
    update_section = f"""
<h2>Data Update Status</h2>
<p><strong>{change_summary}</strong></p>
"""
    if changes:
        update_section += "<ul>\n"
        for change in changes[:20]:
            update_section += f"<li>{change}</li>\n"
        if len(changes) > 20:
            update_section += f"<li>... and {len(changes) - 20} more changes</li>\n"
        update_section += "</ul>\n"
    
    # Part 2: Data summary
    summary_section = f"""
<h2>Data Summary</h2>
<table border="1" cellpadding="5" style="border-collapse: collapse;">
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Data Date</td><td>{date}</td></tr>
<tr><td>Total Combinations</td><td>{stats['total_combos']}</td></tr>
<tr><td>Hardware Platforms</td><td>{len(stats['hardwares'])} types: {', '.join(sorted(stats['hardwares']))}</td></tr>
<tr><td>Models</td><td>{len(stats['models'])} types: {', '.join(sorted(stats['models']))}</td></tr>
<tr><td>Frameworks</td><td>{len(stats['frameworks'])} types: {', '.join(sorted(stats['frameworks']))}</td></tr>
<tr><td>Max Throughput</td><td><strong>{stats['max_tput_overall']:,.2f}</strong> tok/s/GPU<br/>({stats['max_tput_info']})</td></tr>
<tr><td>NVIDIA Best</td><td>{stats['nvidia_best']:,.2f} tok/s/GPU</td></tr>
<tr><td>AMD Best</td><td>{stats['amd_best']:,.2f} tok/s/GPU</td></tr>
</table>

<h3>Peak Performance by Hardware</h3>
<table border="1" cellpadding="5" style="border-collapse: collapse;">
<tr><th>Hardware</th><th>Best Throughput (tok/s/GPU)</th><th>Configuration</th></tr>
"""
    
    # Add best by hardware
    hw_best = defaultdict(lambda: {'tput': 0, 'info': ''})
    for key, d in data.items():
        hw = d['hardware'].upper()
        if d['max_tput'] > hw_best[hw]['tput']:
            hw_best[hw]['tput'] = d['max_tput']
            hw_best[hw]['info'] = f"{d['model']}/{d['framework']}/{d['precision']}"
    
    for hw in sorted(hw_best.keys()):
        summary_section += f"<tr><td>{hw}</td><td>{hw_best[hw]['tput']:,.2f}</td><td>{hw_best[hw]['info']}</td></tr>\n"
    
    summary_section += "</table>\n"
    
    # Add 8k1k detailed table
    table_8k1k = create_8k1k_tables(data_8k1k)
    
    html = f"""
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6;">
<h1>InferenceX Daily Report - {date}</h1>
{update_section}
{summary_section}
{table_8k1k}
<p style="color: #666; font-size: 12px; margin-top: 30px;">
---<br/>
Auto-generated by OpenClaw | Data Source: <a href="https://inferencex.com">InferenceX API</a>
</p>
</body>
</html>
"""
    return html


def send_email(subject, html_content, attachment_path):
    """Send email via SMTP"""
    print(f"Sending email to {EMAIL_TO}...")
    
    # Save email content to file (backup)
    email_file = f"{DATA_DIR}/email_{get_today_date()}.html"
    with open(email_file, 'w') as f:
        f.write(html_content)
    
    print(f"Email content saved to: {email_file}")
    print(f"CSV attachment: {attachment_path}")
    
    # Get SMTP configuration from environment
    smtp_server = os.environ.get('SMTP_SERVER', 'smtp.163.com')
    smtp_port = int(os.environ.get('SMTP_PORT', '465'))
    smtp_user = os.environ.get('SMTP_USER', '')
    smtp_pass = os.environ.get('SMTP_PASSWORD', '')
    
    if not smtp_user or not smtp_pass:
        print("Warning: SMTP credentials not configured. Email not sent.")
        print("Set SMTP_USER and SMTP_PASSWORD environment variables.")
        return False
    
    try:
        # Construct email
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = EMAIL_TO
        msg['Subject'] = subject
        
        # Add HTML content
        msg.attach(MIMEText(html_content, 'html'))
        
        # Add attachment
        with open(attachment_path, 'rb') as f:
            attachment = MIMEBase('application', 'octet-stream')
            attachment.set_payload(f.read())
        encoders.encode_base64(attachment)
        attachment.add_header('Content-Disposition', f'attachment; filename={os.path.basename(attachment_path)}')
        msg.attach(attachment)
        
        # Send via SMTP
        print(f"Connecting to SMTP server {smtp_server}...")
        with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        
        print(f"Email sent successfully! Recipient: {EMAIL_TO}")
        return True
        
    except Exception as e:
        print(f"Email send failed: {e}")
        print(f"Email content saved to: {email_file}")
        return False


def main():
    today = get_today_date()
    yesterday = get_yesterday_date()
    
    print(f"=== InferenceX Daily Report (API Version) - {today} ===\n")
    
    # 1. Get latest data date
    latest_date = get_latest_date_from_api()
    if not latest_date:
        print("Error: Could not get latest data date")
        sys.exit(1)
    
    print(f"\nTarget data date: {latest_date}")
    
    # 2. Fetch all model data from API
    records = fetch_all_models_data(target_date=None)  # Get latest data
    
    if not records:
        print("Error: No data retrieved")
        sys.exit(1)
    
    # 3. Process data
    combo_data, data_8k1k, data_date = process_api_data(records)
    print(f"\nProcessing complete:")
    print(f"  Combinations: {len(combo_data)}")
    print(f"  8k1k combinations: {len(data_8k1k)}")
    print(f"  Data date: {data_date}")
    
    # 4. Generate CSV
    csv_path = f"{DATA_DIR}/inferencex_summary_{today}.csv"
    generate_summary_csv(combo_data, csv_path)
    print(f"\nCSV generated: {csv_path}")
    
    # 5. Save JSON for next comparison
    json_path = f"{DATA_DIR}/inferencex_summary_{today}.json"
    json_data = {}
    for key, d in combo_data.items():
        str_key = str(key)
        json_data[str_key] = {k: (list(v) if isinstance(v, set) else v) for k, v in d.items()}
    with open(json_path, 'w') as f:
        json.dump(json_data, f)
    print(f"JSON saved: {json_path}")
    
    # 6. Load yesterday's data and detect changes
    prev_data_items = load_previous_data(yesterday)
    prev_data = dict(prev_data_items) if prev_data_items else None
    change_summary, changes = detect_changes(combo_data, prev_data)
    print(f"\n{change_summary}")
    
    # 7. Generate statistics
    stats = generate_summary_stats(combo_data)
    
    # 8. Create email
    email_html = create_email_content(data_date, change_summary, changes, stats, combo_data, data_8k1k)
    
    # 9. Send email
    subject = f"[InferenceX] Daily Report - {today} (Data: {data_date})"
    send_email(subject, email_html, csv_path)
    
    print(f"\nDone! Data date: {data_date}")


if __name__ == "__main__":
    main()
