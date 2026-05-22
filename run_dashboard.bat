@echo off
cd /d "%~dp0"
streamlit run app.py --server.address 127.0.0.1
