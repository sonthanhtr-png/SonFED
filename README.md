# SonFED

SonFED là hệ thống nội bộ chạy trên Windows để phân tích XAU/USD, radar vĩ mô Mỹ, tạo chiến lược, gửi cảnh báo Telegram và xuất tín hiệu cho bot MT5 qua file:

`C:\SonFED\shared\signal.json`

Giao diện, kết luận và cảnh báo trong app dùng tiếng Việt.

## 1. Cài Python

Cài Python 3.11 trở lên từ trang chính thức:

https://www.python.org/downloads/windows/

Khi cài đặt, bật tùy chọn `Add python.exe to PATH`.

## 2. Tạo môi trường ảo

Mở PowerShell tại thư mục `sonfed`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 3. Cài thư viện

```powershell
pip install -r requirements.txt
```

## 4. Chạy dashboard

```powershell
streamlit run app.py --server.address 127.0.0.1
```

Dashboard chỉ chạy local trên máy, mặc định tại:

http://localhost:8501

## 5. Cấu hình Telegram

Tạo file `.env` từ `.env.example`:

```env
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

Trong app vào tab `Cài đặt`, bấm `Test Telegram`.

## 6. Cấu hình FRED

Nếu có FRED API key, thêm vào `.env`:

```env
FRED_API_KEY=
```

Nếu không có key, app vẫn chạy bình thường và bỏ qua dữ liệu FRED.

## 7. Cấu hình bot MT5

Bot MT5 đọc tín hiệu tại `C:\SonFED\shared\signal.json` và ghi phản hồi:

- `trade_status.json`
- `risk_status.json`
- `bot_log.json`

Chạy bot:

```powershell
python mt5_trade_bot.py
```

Bot không vào lệnh nếu:

- độ tin cậy dưới ngưỡng
- `allow_auto_trade = false`
- tín hiệu quá cũ
- spread cao
- có tin lớn gần thời điểm hiện tại
- risk check không đạt

Mặc định khối lượng trong bot mẫu là `0.01 lot`. Hãy kiểm tra kỹ tài khoản demo trước khi dùng tài khoản thật.

## 8. Bật app desktop và khay hệ thống

Chạy launcher có icon dưới khay hệ thống:

```powershell
python launcher_tray.py
```

Menu icon cho phép:

- mở dashboard
- tắt chạy nền
- thoát app

## 9. Bật tự khởi động cùng Windows

Chạy PowerShell tại thư mục `sonfed`:

```powershell
.\install_autostart.ps1
```

Script sẽ tạo shortcut trong thư mục Startup của Windows để mở `launcher_tray.py` khi đăng nhập.

## 10. Cấu hình ticker

Ticker nằm trong `config.json` và cũng sửa được trong sidebar dashboard.

Mặc định:

- GOLD: `GC=F`
- DXY: `DX-Y.NYB`
- US10Y: `^TNX`
- US02Y: `^IRX`
- VIX: `^VIX`
- OIL: `CL=F`
- NASDAQ: `QQQ`
- SP500: `SPY`

## 11. Cảnh báo

SonFED không phải lời khuyên đầu tư. Tín hiệu chỉ là hỗ trợ phân tích. Với chế độ tự động, nên thử trên tài khoản demo, giới hạn lot nhỏ và kiểm tra broker/MT5 trước khi chạy thật.
