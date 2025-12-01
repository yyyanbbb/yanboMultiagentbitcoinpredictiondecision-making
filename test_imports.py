# 测试导入脚本
print("=" * 50)
print("测试依赖导入...")
print("=" * 50)

try:
    import numpy as np
    print(f"✅ numpy: {np.__version__}")
except ImportError as e:
    print(f"❌ numpy: {e}")

try:
    import pandas as pd
    print(f"✅ pandas: {pd.__version__}")
except ImportError as e:
    print(f"❌ pandas: {e}")

try:
    from dotenv import load_dotenv
    print("✅ python-dotenv: OK")
except ImportError as e:
    print(f"❌ python-dotenv: {e}")

try:
    from colorama import Fore, Style
    print("✅ colorama: OK")
except ImportError as e:
    print(f"❌ colorama: {e}")

try:
    from openai import OpenAI
    print("✅ openai: OK")
except ImportError as e:
    print(f"❌ openai: {e}")

print("=" * 50)
print("导入测试完成")
print("=" * 50)

