# SonFED Desktop AI Radar

SonFED Desktop dùng PySide6 để bọc lại tinh thần UI của bản Streamlit đã tối ưu: sidebar, tab ngang, AI Decision Box, Radar Forex, Radar FED/Vĩ mô, chart kỹ thuật và Trade Signal Timeline.

Streamlit `app.py` vẫn là bản chính để tham chiếu layout. Desktop không được dùng để xóa hoặc thay thế các tab Streamlit hiện tại.

## Chạy app desktop

```powershell
cd F:\FED\SonFED
pip install -r requirements-desktop.txt
python sonfed_desktop.py
```

SonFED Desktop ghi realtime vào:

```text
C:\SonAI\shared
```

Các file shared giữ nguyên:

- `signal.json`
- `ai_state.json`
- `market_state.json`
- `risk_state.json`
- `heartbeat.json`

## Chạy bản Streamlit chính

```powershell
cd F:\FED\SonFED
streamlit run app.py
```

## Build file exe

```powershell
cd F:\FED\SonFED
pyinstaller --onefile --windowed sonfed_desktop.py
```
