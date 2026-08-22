& .\.venv\Scripts\python.exe -m uvicorn `
  --app-dir .\ai_features-main `
  aifields:app3 `
  --reload `
  --host 0.0.0.0 `
  --port 8000
