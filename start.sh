#!/bin/sh
# Railway / Render / any container host. Pick the app with APP_FILE:
#   streamlit_app.py  -> Discovery Engine dashboard (default)
#   mvp_app.py        -> Memory Detective MVP
exec streamlit run "${APP_FILE:-streamlit_app.py}" \
  --server.port "${PORT:-8501}" --server.address 0.0.0.0 --server.headless true
