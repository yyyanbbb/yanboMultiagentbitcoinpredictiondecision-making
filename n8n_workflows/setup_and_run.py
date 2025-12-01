import os
import sys
import json
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime

# Enable ANSI color codes in Windows
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except:
        pass

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


WORKFLOW_RESULT_FILE = Path(__file__).resolve().parent / "latest_workflow_result.json"
WORKFLOW_RUNS_DIR = Path(__file__).resolve().parent / "workflow_runs"


def save_workflow_snapshot(result: dict):
    """Persist workflow result for GUI consumption"""
    try:
        WORKFLOW_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(WORKFLOW_RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        history_file = WORKFLOW_RUNS_DIR / f"workflow_run_{timestamp}.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"{Colors.GREEN}✓ Workflow snapshot saved to:{Colors.END} {WORKFLOW_RESULT_FILE}")
        print(f"{Colors.GREEN}✓ Run archived as:{Colors.END} {history_file}")
    except Exception as exc:
        print(f"{Colors.RED}Failed to save workflow snapshot: {exc}{Colors.END}")


def print_banner():
    """Print welcome banner"""
    banner = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   {Colors.BOLD}N8N Bitcoin Investment Decision System - Setup & Run{Colors.END}{Colors.CYAN}                       ║
║                                                                              ║
║   Multi-Agent Workflow Automation for Intelligent Investment Decisions       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.END}
"""
    print(banner, flush=True)


def print_n8n_installation_guide():
    """Print N8N installation guide"""
    guide = f"""
{Colors.HEADER}{'='*80}
                        N8N 安装与部署指南
{'='*80}{Colors.END}

{Colors.BOLD}📌 什么是 N8N?{Colors.END}
N8N 是一个开源的工作流自动化工具，可以连接各种应用和服务。
你可以选择以下方式运行 N8N：

{Colors.CYAN}{'─'*80}{Colors.END}

{Colors.BOLD}🌐 方式一：N8N 云服务 (最简单){Colors.END}

1. 访问 https://n8n.io/
2. 点击 "Start Free" 注册账号
3. 创建新工作流
4. 导入 bitcoin_investment_workflow_enhanced.json
5. 配置 HTTP Request 节点指向你的 API 服务器

{Colors.GREEN}优点：无需安装，即开即用
缺点：免费版有执行次数限制{Colors.END}

{Colors.CYAN}{'─'*80}{Colors.END}

{Colors.BOLD}🐳 方式二：Docker 本地部署 (推荐){Colors.END}

# 1. 安装 Docker Desktop
# 下载地址：https://www.docker.com/products/docker-desktop

# 2. 运行 N8N 容器
docker run -it --rm \\
  --name n8n \\
  -p 5678:5678 \\
  -v n8n_data:/home/node/.n8n \\
  n8nio/n8n

# 3. 访问 http://localhost:5678

{Colors.GREEN}优点：完全免费，无限制
缺点：需要安装 Docker{Colors.END}

{Colors.CYAN}{'─'*80}{Colors.END}

{Colors.BOLD}📦 方式三：NPM 全局安装{Colors.END}

# 1. 安装 Node.js (https://nodejs.org/)

# 2. 全局安装 N8N
npm install n8n -g

# 3. 启动 N8N
n8n start

# 4. 访问 http://localhost:5678

{Colors.GREEN}优点：轻量级，启动快
缺点：需要 Node.js 环境{Colors.END}

{Colors.CYAN}{'─'*80}{Colors.END}

{Colors.BOLD}🔧 方式四：使用本项目的一体化方案{Colors.END}

本项目提供了一个完整的解决方案，无需安装 N8N：

1. 启动 API 服务器：
   python n8n_workflows/run_n8n_server.py

2. API 服务器提供了所有工作流节点的功能
3. 你可以直接调用 API 或导入工作流到 N8N

{Colors.YELLOW}推荐：先用方式四测试，确认无误后再部署 N8N{Colors.END}

{Colors.HEADER}{'='*80}{Colors.END}
"""
    print(guide)


def print_workflow_import_guide():
    """Print workflow import guide"""
    guide = f"""
{Colors.HEADER}{'='*80}
                     导入工作流到 N8N
{'='*80}{Colors.END}

{Colors.BOLD}📥 步骤：{Colors.END}

1. 启动 N8N (任选一种方式)
   
2. 打开 N8N 界面 (默认 http://localhost:5678)

3. 点击左上角 "Workflows" → "Import from File"

4. 选择文件：
   {Colors.CYAN}n8n_workflows/bitcoin_investment_workflow_enhanced.json{Colors.END}

5. 配置 HTTP Request 节点：
   - 将所有 http://localhost:5000 改为你的 API 服务器地址
   - 如果在同一台机器上运行，保持默认即可

6. 配置 Slack 节点 (可选)：
   - 添加 Slack 凭据
   - 或删除 Slack 节点改用其他通知方式

7. 点击 "Save" 保存工作流

8. 点击右上角开关激活工作流

{Colors.BOLD}🧪 测试工作流：{Colors.END}

方式一：手动执行
- 点击 "Execute Workflow" 按钮

方式二：Webhook 触发
- 复制 Webhook URL
- 发送 POST 请求：
  curl -X POST http://localhost:5678/webhook/bitcoin-analysis

方式三：定时执行
- 工作流默认每 6 小时自动执行一次

{Colors.HEADER}{'='*80}{Colors.END}
"""
    print(guide)


def check_api_server(base_url="http://localhost:5000", timeout=3):
    """Check if API server is running"""
    try:
        import requests
        response = requests.get(f"{base_url}/api/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False


def ensure_api_server_running():
    """Ensure API server is running, prompt user if not"""
    if check_api_server():
        return True
    
    print(f"\n{Colors.RED}⚠️  API Server is not running!{Colors.END}")
    print(f"{Colors.YELLOW}The API server must be running before testing or running workflows.{Colors.END}\n")
    
    print(f"{Colors.BOLD}Options:{Colors.END}")
    print(f"  1. Start API server in another terminal window")
    print(f"     Command: {Colors.CYAN}python n8n_workflows/run_n8n_server.py{Colors.END}")
    print(f"  2. Start API server automatically (will run in background)")
    print(f"  3. Return to main menu\n")
    
    try:
        choice = input(f"{Colors.CYAN}Enter your choice (1-3): {Colors.END}").strip()
        
        if choice == "2":
            print(f"\n{Colors.YELLOW}Starting API server in background...{Colors.END}")
            print(f"{Colors.YELLOW}Note: The server will run in a separate process.{Colors.END}")
            print(f"{Colors.YELLOW}You can stop it by closing the new window or using Ctrl+C.{Colors.END}\n")
            
            # Get paths
            script_dir = Path(__file__).resolve().parent
            server_script = script_dir / "run_n8n_server.py"
            project_root = script_dir.parent
            
            if sys.platform == 'win32':
                # Windows: Start in new console window
                import subprocess
                try:
                    subprocess.Popen(
                        [sys.executable, str(server_script)],
                        cwd=str(project_root),
                        creationflags=subprocess.CREATE_NEW_CONSOLE
                    )
                except AttributeError:
                    # Fallback for older Python versions
                    subprocess.Popen(
                        [sys.executable, str(server_script)],
                        cwd=str(project_root)
                    )
            else:
                # Unix/Linux: Start in background
                subprocess.Popen(
                    [sys.executable, str(server_script)],
                    cwd=str(project_root),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            # Wait for server to start
            print(f"{Colors.CYAN}Waiting for server to start...{Colors.END}")
            for i in range(10):
                time.sleep(1)
                if check_api_server():
                    print(f"{Colors.GREEN}✓ API Server is now running!{Colors.END}\n")
                    return True
                print(f"  Attempt {i+1}/10...", end='\r')
            
            print(f"\n{Colors.RED}Server did not start in time. Please start it manually.{Colors.END}")
            return False
        elif choice == "3":
            return False
        else:
            print(f"\n{Colors.YELLOW}Please start the API server manually and try again.{Colors.END}")
            return False
            
    except (EOFError, KeyboardInterrupt):
        return False


def test_api_endpoints():
    """Test all API endpoints"""
    print(f"\n{Colors.BOLD}🧪 Testing API Endpoints...{Colors.END}\n")
    
    # Check if server is running
    if not check_api_server():
        if not ensure_api_server_running():
            return False
    
    try:
        import requests
    except ImportError:
        print(f"{Colors.YELLOW}Installing requests...{Colors.END}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
        import requests
    
    base_url = "http://localhost:5000"
    workflow_data = {
        "timestamp": datetime.now().isoformat()
    }
    
    tests = [
        ("Health Check", "GET", "/api/health", None),
        ("Fetch Price", "POST", "/api/fetch-price", {"symbol": "BTC/USD"}),
        ("Calculate Indicators", "POST", "/api/calculate-indicators", {"price_data": {"price_series": [67000 + i*100 for i in range(100)]}}),
        ("Linear Regression", "POST", "/api/linear-regression-predict", {"current_price": 67500, "features": {}}),
        ("ARIMA Predict", "POST", "/api/arima-predict", {"price_series": [67000 + i*100 for i in range(100)], "steps": 7}),
        ("Seasonal Predict", "POST", "/api/seasonal-predict", {"values": [67000 + i*100 for i in range(100)], "steps": 7}),
        ("Multi-Timeframe", "POST", "/api/multi-timeframe-analysis", {"prediction": {"change_percent": 5.2}}),
        ("Scenario Analysis", "POST", "/api/scenario-analysis", {"current_price": 67500, "prediction": {"change_percent": 5.2}}),
        ("Technical Analyst", "POST", "/api/analyst/technical", {"prediction": {"change_percent": 5.2}}),
        ("Devils Advocate", "POST", "/api/devils-advocate", {"consensus_view": "BUY (3/5)"}),
        ("Red-Blue Team", "POST", "/api/red-blue-team", {"decision": {"action": "buy", "confidence": 7}}),
        ("Bayesian Fusion", "POST", "/api/bayesian-fusion", {"signals": {}}),
        ("Confidence Calibration", "POST", "/api/confidence-calibration", {"raw_confidence": 0.72}),
        ("Chairman Synthesis", "POST", "/api/chairman-synthesis", {"rankings": [], "devils_advocate": {"confidence_in_challenge": 6}, "red_blue_verdict": {"final_score": 7}, "bayesian_result": {"up_probability": 0.68}, "calibrated_confidence": 0.65}),
        ("Risk Budget", "POST", "/api/risk-budget-calculation", {"total_capital": 100000, "entry_price": 67500, "confidence": 0.7}),
    ]
    
    results = []
    
    for name, method, endpoint, data in tests:
        try:
            if method == "GET":
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
            else:
                response = requests.post(f"{base_url}{endpoint}", json=data, timeout=30)
            
            if response.status_code == 200:
                status = f"{Colors.GREEN}✓ PASS{Colors.END}"
                results.append((name, True))
            else:
                status = f"{Colors.RED}✗ FAIL ({response.status_code}){Colors.END}"
                results.append((name, False))
        except Exception as e:
            status = f"{Colors.RED}✗ Error: {str(e)[:30]}{Colors.END}"
            results.append((name, False))
        
        print(f"  {status} - {name}")
    
    # Summary
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    print(f"\n{Colors.BOLD}Summary: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"{Colors.GREEN}All API endpoints are working correctly!{Colors.END}")
    else:
        print(f"{Colors.YELLOW}Some endpoints failed. Make sure the API server is running.{Colors.END}")
    
    return passed == total


def run_full_workflow_test():
    """Run a complete workflow simulation"""
    print(f"\n{Colors.BOLD}🔄 Running Full Workflow Simulation...{Colors.END}\n")
    
    # Check if server is running
    if not check_api_server():
        if not ensure_api_server_running():
            print(f"\n{Colors.RED}Cannot run workflow test without API server.{Colors.END}")
            return False
    
    try:
        import requests
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
        import requests
    
    base_url = "http://localhost:5000"
    workflow_data = {
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Step 1: Fetch price
        print(f"{Colors.CYAN}Step 1: Fetching Bitcoin price...{Colors.END}")
        resp = requests.post(f"{base_url}/api/fetch-price", json={"symbol": "BTC/USD"}, timeout=30)
        price_data = resp.json()
        current_price = price_data.get("current_price", 67500)
        workflow_data["price_data"] = price_data
        print(f"  Current Price: ${current_price:,.2f}")
        
        # Step 2: Calculate indicators
        print(f"{Colors.CYAN}Step 2: Calculating technical indicators...{Colors.END}")
        resp = requests.post(f"{base_url}/api/calculate-indicators", json={"price_data": price_data}, timeout=30)
        indicators = resp.json()
        workflow_data["technical_indicators"] = indicators
        print(f"  RSI: {indicators.get('technical_indicators', {}).get('rsi_14', 'N/A'):.2f}")
        print(f"  MACD: {indicators.get('technical_indicators', {}).get('macd', 'N/A'):.4f}")
        
        # Step 3: Model predictions
        print(f"{Colors.CYAN}Step 3: Running prediction models...{Colors.END}")
        
        lr_resp = requests.post(f"{base_url}/api/linear-regression-predict", 
                               json={"current_price": current_price, "features": indicators.get("features", {})}, timeout=30)
        lr_result = lr_resp.json()
        print(f"  Linear Regression: ${lr_result.get('predicted_price', 0):,.2f} ({lr_result.get('change_percent', 0):+.2f}%)")
        
        arima_resp = requests.post(f"{base_url}/api/arima-predict",
                                  json={"price_series": price_data.get("price_series", []), "steps": 7}, timeout=30)
        arima_result = arima_resp.json()
        print(f"  ARIMA: ${arima_result.get('predicted_price', 0):,.2f} ({arima_result.get('change_percent', 0):+.2f}%)")
        
        seasonal_resp = requests.post(f"{base_url}/api/seasonal-predict",
                                     json={"values": price_data.get("price_series", []), "steps": 7}, timeout=30)
        seasonal_result = seasonal_resp.json()
        print(f"  Seasonal: ${seasonal_result.get('predicted_price', 0):,.2f} ({seasonal_result.get('change_percent', 0):+.2f}%)")
        
        # Calculate ensemble
        ensemble_price = (lr_result.get('predicted_price', current_price) * 0.5 +
                         arima_result.get('predicted_price', current_price) * 0.3 +
                         seasonal_result.get('predicted_price', current_price) * 0.2)
        ensemble_change = ((ensemble_price - current_price) / current_price) * 100
        
        workflow_data["model_predictions"] = {
            "linear_regression": lr_result,
            "arima": arima_result,
            "seasonal": seasonal_result,
            "ensemble": {
                "predicted_price": ensemble_price,
                "change_percent": ensemble_change,
                "current_price": current_price
            }
        }
        
        print(f"{Colors.GREEN}  Ensemble: ${ensemble_price:,.2f} ({ensemble_change:+.2f}%){Colors.END}")
        
        # Step 4: Advanced analysis
        print(f"{Colors.CYAN}Step 4: Running advanced analysis...{Colors.END}")
        prediction = {"change_percent": ensemble_change, "predicted_price": ensemble_price}
        
        mtf_resp = requests.post(f"{base_url}/api/multi-timeframe-analysis", json={"prediction": prediction}, timeout=30)
        mtf_result = mtf_resp.json()
        workflow_data["multi_timeframe_analysis"] = mtf_result
        print(f"  Multi-Timeframe Signal: {mtf_result.get('analysis', {}).get('synthesis', {}).get('final_signal', 'N/A')}")
        
        scenario_resp = requests.post(f"{base_url}/api/scenario-analysis", 
                                     json={"current_price": current_price, "prediction": prediction}, timeout=30)
        scenario_result = scenario_resp.json()
        workflow_data["scenario_analysis"] = scenario_result
        expected_change = scenario_result.get('scenarios', {}).get('expected_value', {}).get('expected_change', 0)
        print(f"  Expected Value: {expected_change:+.2f}%")
        
        # Step 5: Analyst opinions
        print(f"{Colors.CYAN}Step 5: Gathering analyst opinions...{Colors.END}")
        analysts = ["technical", "industry", "financial", "market", "risk"]
        opinions = []
        for analyst in analysts:
            resp = requests.post(f"{base_url}/api/analyst/{analyst}", json={"prediction": prediction}, timeout=30)
            result = resp.json()
            rec = result.get('analysis', {}).get('recommendation', 'hold')
            conf = result.get('analysis', {}).get('confidence', 5)
            opinions.append({
                "type": analyst,
                "recommendation": rec,
                "confidence": conf,
                "analysis": result.get('analysis', {}),
                "meta": result
            })
            print(f"  {analyst.capitalize()}: {rec.upper()} ({conf}/10)")
        
        workflow_data["analyst_opinions"] = opinions
        
        # Step 6: Peer review
        print(f"{Colors.CYAN}Step 6: Conducting peer review...{Colors.END}")
        anonymized_payload = []
        for o in opinions:
            anonymized_payload.append({
                "analyst": {"type": o["type"]},
                "analysis": {
                    "recommendation": o["recommendation"],
                    "confidence": o["confidence"],
                    "key_points": o.get("analysis", {}).get("key_points", []),
                    "content": o.get("analysis", {}).get("content", "")
                }
            })
        
        anon_resp = requests.post(
            f"{base_url}/api/anonymize-analyses",
            json={"analyses": anonymized_payload},
            timeout=30
        )
        anon_result = anon_resp.json()
        workflow_data["anonymous_analyses"] = anon_result
        
        peer_resp = requests.post(
            f"{base_url}/api/peer-review",
            json={"anonymous_analyses": anon_result.get("anonymous_analyses", [])},
            timeout=30
        )
        peer_result = peer_resp.json()
        workflow_data["peer_review"] = peer_result
        print(f"  Total Reviews: {peer_result.get('total_reviews', 0)}")
        
        rank_resp = requests.post(
            f"{base_url}/api/calculate-rankings",
            json={
                "peer_reviews": peer_result.get("peer_reviews", []),
                "anonymous_analyses": anon_result.get("anonymous_analyses", [])
            },
            timeout=30
        )
        rank_result = rank_resp.json()
        workflow_data["rankings"] = rank_result
        consensus = rank_result.get('consensus', 'N/A')
        workflow_data["analyst_consensus"] = consensus
        print(f"  Consensus: {consensus}")
        
        # Step 7: Validation
        print(f"{Colors.CYAN}Step 7: Running validation...{Colors.END}")
        
        da_resp = requests.post(
            f"{base_url}/api/devils-advocate",
            json={"consensus_view": consensus},
            timeout=30
        )
        da_result = da_resp.json()
        workflow_data["devils_advocate"] = da_result
        da_conf = da_result.get('devils_advocate', {}).get('confidence_in_challenge', 5)
        print(f"  Devil's Advocate Challenge: {da_conf}/10")
        
        rb_resp = requests.post(
            f"{base_url}/api/red-blue-team",
            json={"decision": {"action": "buy" if ensemble_change > 0 else "hold", "confidence": 7}},
            timeout=30
        )
        rb_result = rb_resp.json()
        workflow_data["red_blue_team"] = rb_result
        verdict = rb_result.get('verdict', {}).get('verdict', 'N/A')
        print(f"  Red-Blue Verdict: {verdict}")
        
        # Step 8: Final decision
        print(f"{Colors.CYAN}Step 8: Making final decision...{Colors.END}")
        
        bf_resp = requests.post(f"{base_url}/api/bayesian-fusion", json={"signals": {}}, timeout=30)
        bf_result = bf_resp.json()
        workflow_data["bayesian_fusion"] = bf_result
        
        cc_resp = requests.post(f"{base_url}/api/confidence-calibration", json={"raw_confidence": 0.72}, timeout=30)
        cc_result = cc_resp.json()
        workflow_data["confidence_calibration"] = cc_result
        calibrated = cc_result.get('calibration', {}).get('calibrated_confidence', 0.65)
        
        chairman_resp = requests.post(f"{base_url}/api/chairman-synthesis", json={
            "rankings": rank_result.get('rankings', []),
            "devils_advocate": da_result.get('devils_advocate', {}),
            "red_blue_verdict": rb_result.get('verdict', {}),
            "bayesian_result": bf_result.get('fusion_result', {}),
            "calibrated_confidence": calibrated,
            "consensus": rank_result.get('consensus'),
            "position_guidance": {
                "stop_loss_pct": 8,
                "take_profit_pct": 12
            }
        }, timeout=30)
        chairman_result = chairman_resp.json()
        workflow_data["chairman_synthesis"] = chairman_result
        
        final_action = chairman_result.get('chairman_decision', {}).get('action', 'hold')
        final_position = chairman_result.get('chairman_decision', {}).get('position_percent', 0)
        
        # Step 9: Risk budget
        risk_resp = requests.post(f"{base_url}/api/risk-budget-calculation", json={
            "total_capital": 100000,
            "entry_price": current_price,
            "confidence": calibrated
        }, timeout=30)
        risk_result = risk_resp.json()
        workflow_data["risk_budget"] = risk_result
        
        # Final output
        print(f"\n{Colors.HEADER}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}                    FINAL DECISION{Colors.END}")
        print(f"{Colors.HEADER}{'='*60}{Colors.END}")
        print(f"""
  📊 Current Price:     ${current_price:,.2f}
  📈 Predicted Price:   ${ensemble_price:,.2f} ({ensemble_change:+.2f}%)
  
  🎯 Action:            {Colors.GREEN if 'buy' in final_action else Colors.YELLOW}{final_action.upper()}{Colors.END}
  💰 Position:          {final_position}%
  🔒 Confidence:        {calibrated:.1%}
  
  📉 Stop Loss:         ${current_price * 0.92:,.2f} (-8%)
  📈 Take Profit:       ${current_price * 1.12:,.2f} (+12%)
  
  💵 Investment Amount: ${risk_result.get('risk_budget', {}).get('total_investment', 0):,.2f}
  ⚠️  Max Loss:          ${risk_result.get('risk_budget', {}).get('max_loss', 0):,.2f}
""")
        print(f"{Colors.HEADER}{'='*60}{Colors.END}")
        
        # Save decision
        save_resp = requests.post(f"{base_url}/api/save-decision", json={
            "decision": {
                "timestamp": datetime.now().isoformat(),
                "current_price": current_price,
                "predicted_price": ensemble_price,
                "action": final_action,
                "position": final_position,
                "confidence": calibrated
            }
        }, timeout=30)
        save_result = save_resp.json()
        workflow_data["decision_saved_to"] = save_result.get('saved_to')
        print(f"\n{Colors.GREEN}Decision saved to: {save_result.get('saved_to', 'N/A')}{Colors.END}")
        
        workflow_data["summary"] = {
            "current_price": current_price,
            "ensemble_price": ensemble_price,
            "ensemble_change_percent": ensemble_change,
            "final_action": final_action,
            "final_position_percent": final_position,
            "calibrated_confidence": calibrated,
            "risk_budget": risk_result.get('risk_budget', {}),
            "decision_file": save_result.get('saved_to')
        }
        
        save_workflow_snapshot(workflow_data)
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n{Colors.RED}Error: {error_msg}{Colors.END}")
        
        # Check if it's a connection error
        if "Connection" in error_msg or "refused" in error_msg.lower() or "10061" in error_msg:
            print(f"\n{Colors.YELLOW}The API server appears to have stopped.{Colors.END}")
            print(f"{Colors.YELLOW}Please restart it: {Colors.CYAN}python n8n_workflows/run_n8n_server.py{Colors.END}")
        else:
            print(f"{Colors.YELLOW}Please check the error message above and try again.{Colors.END}")
        
        return False


def start_api_server():
    """Start the API server"""
    script_dir = Path(__file__).resolve().parent
    server_script = script_dir / "n8n_api_server.py"
    
    print(f"\n{Colors.BOLD}🚀 Starting API Server...{Colors.END}\n")
    
    # Add project root to path
    project_root = script_dir.parent
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    
    # Check dependencies
    try:
        import flask
        import flask_cors
    except ImportError:
        print(f"{Colors.YELLOW}Installing required packages...{Colors.END}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "flask-cors", "-q"])
    
    # Run server
    subprocess.run([sys.executable, str(server_script)])


def show_menu():
    """Show interactive menu"""
    while True:
        print(f"""
{Colors.BOLD}📋 Main Menu{Colors.END}

  1. 📖 Show N8N Installation Guide
  2. 📥 Show Workflow Import Guide  
  3. 🚀 Start API Server
  4. 🧪 Test API Endpoints
  5. 🔄 Run Full Workflow Simulation
  6. ❌ Exit

""", flush=True)
        try:
            choice = input(f"{Colors.CYAN}Enter your choice (1-6): {Colors.END}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{Colors.GREEN}Goodbye!{Colors.END}\n")
            break
        
        if choice == "1":
            print_n8n_installation_guide()
        elif choice == "2":
            print_workflow_import_guide()
        elif choice == "3":
            start_api_server()
            break
        elif choice == "4":
            test_api_endpoints()
        elif choice == "5":
            run_full_workflow_test()
        elif choice == "6":
            print(f"\n{Colors.GREEN}Goodbye!{Colors.END}\n")
            break
        else:
            print(f"{Colors.RED}Invalid choice. Please try again.{Colors.END}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")


def main():
    """Main entry point"""
    # Force UTF-8 encoding for Windows
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            pass
    
    parser = argparse.ArgumentParser(description="N8N Bitcoin Investment System Setup")
    parser.add_argument("--server", action="store_true", help="Start API server")
    parser.add_argument("--test", action="store_true", help="Test API endpoints")
    parser.add_argument("--workflow", action="store_true", help="Run full workflow simulation")
    parser.add_argument("--install", action="store_true", help="Show N8N installation guide")
    
    args = parser.parse_args()
    
    print_banner()
    sys.stdout.flush()
    
    if args.server:
        start_api_server()
    elif args.test:
        test_api_endpoints()
    elif args.workflow:
        run_full_workflow_test()
    elif args.install:
        print_n8n_installation_guide()
        print_workflow_import_guide()
    else:
        show_menu()


if __name__ == "__main__":
    main()