@echo off
echo Starting AI Voice Sales Agent (Port 8001)...
cd /d "E:\project\ceo 2.0\cli helper"
call .venv\Scripts\activate
cd sales_agent
python server.py
pause
