import subprocess
import sys

print("Installing statsmodels...")
result = subprocess.run([sys.executable, "-m", "pip", "install", "statsmodels"], 
                       capture_output=True, text=True)
print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
print("Return code:", result.returncode)

print("\nVerifying installation...")
try:
    import statsmodels
    print(f"statsmodels version: {statsmodels.__version__}")
except ImportError as e:
    print(f"Import failed: {e}")

