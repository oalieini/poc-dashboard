# xAPI Learning Analytics Dashboard

A full-stack, dynamic dashboard for **xAPI (Tin Can)** learning analytics.  
Connects to any LRS (Learning Record Store) via REST API and computes a catalog of learning indicators.

---

## Quick Start

### 1. Install & run

```bash
git clone https://github.com/oalieini/poc-dashboard.git
cd poc-dashboard
pip install -r requirements.txt
streamlit run frontend/app.py
```

Open: [http://localhost:8501](http://localhost:8501)  

### 2. Connect to a real LRS

```bash
cp .env.example .env
# Edit .env with your LRS credentials
streamlit run frontend/app.py
```

Then toggle off "Use Mock / Demo Data" in the sidebar.

### 3. Docker

```bash
docker-compose up
```

---
## Configuration

### Auth Methods

**Basic Auth**:
```env
LRS_ENDPOINT=https://lrs.example.com/xapi
LRS_USERNAME=admin
LRS_PASSWORD=secret
```

---


