from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.api import app as api_app

app = FastAPI(title="M5 Forecasting App")
app.mount("/api", api_app)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <head>
        <title>M5 Forecasting Demo</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 40px; }
          .card { border: 1px solid #ddd; padding: 20px; border-radius: 8px; max-width: 700px; }
          a { color: #2563eb; text-decoration: none; }
        </style>
      </head>
      <body>
        <div class="card">
          <h1>M5 Forecasting Demo</h1>
          <p>This deployment hosts the forecasting API and a simple frontend landing page.</p>
          <p>Use the API endpoint at <a href="/api/health">/api/health</a> or <a href="/api/docs">/api/docs</a>.</p>
          <p>For the full Streamlit interface, run it locally with <code>streamlit run app/streamlit_app.py</code>.</p>
        </div>
      </body>
    </html>
    """
