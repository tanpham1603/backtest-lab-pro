import subprocess
import sys
import os

def main():
    """Script khởi động dashboard"""
    print("🚀 Starting Backtest Lab Pro...")
    
    # Kiểm tra dependencies
    try:
        import streamlit
        import vectorbt
        import plotly
        print("✅ Dependencies check passed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Chạy Streamlit
    main_app = os.path.join("app", "main.py")
    
    if os.path.exists(main_app):
        print(f"🌐 Starting dashboard at http://localhost:8501")
        subprocess.run(["streamlit", "run", main_app])
    else:
        print("❌ Main app not found. Please check file structure.")

if __name__ == "__main__":
    main()
