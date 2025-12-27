# NLP Inference Microservice (Docker Compose & Redis)

![CI Pipeline](https://github.com/Western-1/nlp-inference-service/actions/workflows/ci.yml/badge.svg)
![CD Pipeline](https://github.com/Western-1/nlp-inference-service/actions/workflows/deploy.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.9-blue.svg) 
![License](https://img.shields.io/badge/license-MIT-green.svg) 
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white) 
![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?style=flat&logo=redis&logoColor=white) 
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)
![Nginx](https://img.shields.io/badge/nginx-%23009639.svg?style=flat&logo=nginx&logoColor=white)
![SSL](https://img.shields.io/badge/SSL-Secure-green.svg?style=flat&logo=letsencrypt)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=flat&logo=Prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-F46800?style=flat&logo=Grafana&logoColor=white)
![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=flat&logo=WeightsAndBiases&logoColor=white)
![AWS S3](https://img.shields.io/badge/AWS_S3-FF9900?style=flat&logo=amazons3&logoColor=white)
![Boto3](https://img.shields.io/badge/Boto3-Cloud_Storage-blue)

A production-ready **Microservices Architecture** for Natural Language Processing. This project orchestrates multiple containers using **Docker Compose**: a FastAPI application for inference and a **Redis** database for high-speed logging and persistence.

It features a fully automated **CI/CD Pipeline** via GitHub Actions.

![HTTPS Link](Images/HTTPS.png)

![Dashboard Overview](Images/1.png)

## Architecture & Workflow

This project demonstrates a modern microservices approach. Instead of a monolithic script, the system decouples inference from data persistence and includes automated testing pipelines.

```mermaid
graph TD
  %% --- Styling Definitions ---
  classDef app fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:#000
  classDef db fill:#ffcdd2,stroke:#c62828,stroke-width:2px,color:#000
  classDef ext fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
  classDef proxy fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
  classDef monitor fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#000
  classDef alert fill:#ffab91,stroke:#d84315,stroke-width:2px,color:#000

  %% --- Actors ---
  User([User / Client])
  style User fill:#fff,stroke:#333,stroke-width:2px,color:#000

  %% --- Ingress Layer (Production) ---
  subgraph Ingress [Ingress Layer]
    Nginx[Nginx Reverse Proxy<br/>SSL Termination]:::proxy
  end

  %% --- Private Docker Network ---
  subgraph DockerNet [Private Docker Network]
    
    %% App Service
    subgraph Container_App [App Service]
      Gunicorn[Gunicorn Manager]:::app
      subgraph Workers [Async Workers]
        Uvicorn[Uvicorn Worker]:::app
        Logic[ML Inference Logic]:::app
      end
    end

    %% Archiver & Storage
    Archiver[Python Archiver Service]:::app
    Redis[(Redis DB)]:::db

    %% Monitoring Stack
    subgraph Observability [Monitoring Stack]
        Prometheus[Prometheus]:::monitor
        Grafana[Grafana Dashboards]:::monitor
        Alertmanager[Alertmanager]:::alert
    end
  end

  %% --- External / Cloud ---
  subgraph External [External Resources]
    HF_Hub[HuggingFace Hub]:::ext
    HFCache[Volume: HF Cache]:::ext
    S3[(AWS S3 Archive)]:::ext
    Telegram([Telegram Bot]):::ext
  end

  %% === Data Flow ===
  User -->|"1. HTTPS (Port 443)"| Nginx
  Nginx -->|"2. Proxy Pass (Port 8000)"| Gunicorn

  Gunicorn -->|"3. Spawn Processes"| Uvicorn
  Uvicorn -->|"4. Inference"| Logic
  
  Logic -.->|"Download (First run)"| HF_Hub
  Logic -->|"Load from"| HFCache
  
  Uvicorn -->|"5. LPUSH (Async logs)"| Redis
  Redis -->|"LTRIM (Auto-cleanup)"| Redis
  
  User -->|"GET /history"| Uvicorn
  Uvicorn <-->|"LRANGE"| Redis

  %% Archiver / Data Pipeline (Active)
  Archiver -->|"1. Fetch & Clear"| Redis
  Archiver -->|"2. Upload JSON"| S3

  %% Monitoring Flow
  Prometheus -->|"Scrape /metrics"| Uvicorn
  Grafana -->|"Query Data"| Prometheus

  %% Alerting Flow
  Prometheus -.->|"Fire Alert (Down > 1m)"| Alertmanager
  Alertmanager -.->|"Send Notification"| Telegram
  Telegram -.->|"Critical Alert"| User

  style DockerNet fill:none,stroke:#607d8b,stroke-width:2px,stroke-dasharray: 5 5
  style Ingress fill:none,stroke:none
  style External fill:none,stroke:none
  style Container_App fill:#f1f8e9,stroke:#558b2f,stroke-width:1px
  style Workers fill:#fff,stroke:none
```

## Key Features

- **Microservices Orchestration:** Fully dockerized environment via `docker-compose`.
- **CI/CD Pipeline:** Automated testing via GitHub Actions and **Automatic Deployment to AWS EC2** on every push to `main`.
- **Multi-Model Inference:** `DistilBERT` (Sentiment) & `Helsinki-NLP` (Translation).
- **Persistent Storage:** Asynchronous logging to Redis using `LPUSH`/`LTRIM`.
- **Mocked Testing:** Unit tests use `unittest.mock` to simulate ML models and Redis in CI environments.
- **MLOps Integration:** Real-time experiment tracking and model performance monitoring via **Weights & Biases**.

## Data Retention & Archiving

To prevent Redis memory exhaustion, the system implements a **scheduled archiving pipeline**:
- **Archiver Service:** A lightweight Python container that runs on a schedule.
- **Workflow:** 1. Every 60 seconds, it performs an atomic `RENAME` of the log key in Redis.
  2. It converts the raw data into a structured `.json` file.
  3. The file is uploaded to an **AWS S3 Bucket** with a timestamped filename.
  4. Local cache and temporary Redis keys are cleared.
- **Benefits:** Long-term storage for ML re-training while keeping the production DB lean.

![Archive console](Images/Archive_console.png)

## Tech Stack

- **Orchestration:** Docker Compose
- **CI/CD:** GitHub Actions
- **Core:** Python 3.9, FastAPI, Uvicorn
- **Database:** Redis (Alpine)
- **ML Backend:** PyTorch, Transformers
- **Infrastructure:** Docker Compose, Nginx (Reverse Proxy)
- **Security:** SSL/TLS (Let's Encrypt), Automated Cert Renewal
- **Models:**
  - `distilbert-base-uncased-finetuned-sst-2-english`
  - `Helsinki-NLP/opus-mt-en-fr`

## Project Structure

```
.
├── app/                  # Inference Service (FastAPI)
│   ├── main.py           # API endpoints and ML logic
│   ├── Dockerfile        # Multi-stage production build
│   └── requirements.txt  # NLP & Web dependencies
├── archiver/             # Data Pipeline Service (S3 Worker)
│   ├── main.py           # Scheduled archiving logic
│   ├── Dockerfile        # Lightweight Python environment
│   └── requirements.txt  # Boto3, Redis, and Scheduling tools
├── alertmanager/         # Monitoring alerts configuration
├── grafana/              # Dashboards as Code (JSON)
├── nginx/                # Reverse Proxy & SSL configuration
├── prometheus/           # Metrics collection & alerting rules
├── tests/                # Unit & Integration testing suite
├── docker-compose.yml    # Full stack orchestration
├── .github/workflows/    # CI/CD Automated Pipelines
└── README.md             # Documentation

```

## Installation and Setup

### Prerequisites
Docker Engine & Docker Compose installed.

### Quick Start (Local)
1. **Clone the repository:**
```bash
git clone https://github.com/Western-1/nlp-inference-service
cd nlp-inference-service
```
2. **Start the Services:**
```bash
docker-compose up --build
```
3. **Access API:**
Open `http://localhost:8000/docs` to see the Swagger UI.

4. **Create .env file:**
Create a `.env` file in the root directory with your credentials (optional for local test, required for S3):
```bash
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET_NAME=your_backet_name
SERVER_API_KEY=demo
...
```
and etc..

## Development & Testing

This project uses **Pytest** for unit and integration testing. The CI pipeline runs these tests automatically.

To run tests locally:
```bash
pip install pytest httpx

PYTHONPATH=. pytest tests/ -v
```

## Code Quality & Security
The project follows strict development standards to ensure code quality and security. Every commit triggers a GitHub Actions pipeline that runs:

- Linting (`flake8`): Enforces PEP8 style guide and catches syntax errors.

- Security Scanning (`bandit`): Scans the code for common vulnerabilities (e.g., hardcoded secrets, unsafe functions).

## Run checks locally
You can run the same checks on your machine before pushing code:
```bash
# 1. Install quality tools
pip install flake8 bandit

# 2. Check code style (Linting)
flake8 . --count --show-source --statistics

# 3. Scan for security vulnerabilities
bandit -r .
```

## Deployment (AWS EC2)

The project uses a **Continuous Deployment (CD)** pipeline. Any change pushed to the `main` branch is automatically deployed to the AWS EC2 instance using GitHub Actions.

### 1. Initial Setup (One-time)
Provision the infrastructure and setup Docker:
1. Launch an AWS `t3.micro` instance (Ubuntu 24.04).
2. Configure Security Group: Open ports `22` (SSH), `80` (HTTP), and `443` (HTTPS).
3. Connect via SSH and install Docker & Docker Compose.
4. Clone the repo manually **only for the first run**:
   ```bash
   git clone [https://github.com/Western-1/nlp-inference-service](https://github.com/Western-1/nlp-inference-service)
   cd nlp-inference-service
   docker compose up -d --build

### 2. Configure GitHub Secrets
For the CD pipeline to work, add these secrets in repo settings (`Settings` -> `Secrets and variables` -> `Actions`):

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `EC2_HOST` | Public IP / DNS | Address of your AWS instance |
| `EC2_USER` | `ubuntu` | SSH Username |
| `EC2_SSH_KEY` | `-----BEGIN RSA...` | Private SSH Key content |
| `WANDB_API_KEY`| `ef2f...` | API Key for Weights & Biases |
| `TELEGRAM_TOKEN`| `12345:ABC...` | Bot Token from @BotFather |
| `TELEGRAM_CHAT_ID`| `12345678` | Your User ID for notifications |
| `AWS_ACCESS_KEY_ID` | `AKIA...` | IAM User Access Key with S3 permissions |
| `AWS_SECRET_ACCESS_KEY` | `wJalrX...` | IAM User Secret Access Key |
| `S3_BUCKET_NAME` | `western-nlp-logs-archive` | Target S3 Bucket name |
| `DEMO_KEY` | `demo` | **Required.** API Key injected as SERVER_API_KEY |

## 3. Automatic Updates
No manual action is required for updates.

Push changes to `main`.

GitHub Actions will SSH into the server, pull the latest code, rebuild containers, and cleanup unused images.

## 🔐 Authentication

To prevent unauthorized usage, the API implements **API Key Authentication**.
All inference and history endpoints (`/sentiment`, `/translate`, `/history`) require the `X-API-Key` header. Public endpoints (`/`, `/health`) remain open.

### 🔑 Public Demo Key
For recruitment and testing purposes, a **public demo key** is available:

> **Header Name:** `X-API-Key`
>
> **Demo Value:** `demo`

*(Please use this key responsibly. It is rate-limited and monitored.)*

**Example cURL Request:**
```bash
curl -X POST "[https://western-nlp.ddns.net/sentiment](https://western-nlp.ddns.net/sentiment)" \
     -H "X-API-Key: demo" \
     -H "Content-Type: application/json" \
     -d '{"text": "Security implementation is crucial for MLOps."}'

## API Documentation

### 1. Health Check
`GET /` - Checks service status and Redis connection.

### 2. Request History
`GET /history` - Returns the last 10 requests stored in Redis.
![History Example](Images/3.png)

### 3. Sentiment Analysis
`POST /sentiment` - Classifies text as **POSITIVE** or **NEGATIVE**.

**Example Request:**
```json
{
  "text": "The deployment process was incredibly smooth."
}
```
**Example Response:**
```json
{
  "result": [
    {
      "label": "POSITIVE",
      "score": 0.9998
    }
  ]
}
```

### 4. Translation (En → Fr)
`POST /translate` - Translates English text to French.

**Example Response:**
```json
{
  "translated_text": "Bonjour le monde, c'est un test."
}
```

![Translation Example](Images/2.png)

## Live Demo

Try the API live here (Reverse Proxy via Nginx):  
**[https://western-nlp.ddns.net/docs](https://western-nlp.ddns.net/docs)**

> [!IMPORTANT]
> Active (Hosted on AWS EC2 Free Tier)

> [!NOTE]
> Since this is a free tier instance, it might be stopped to save resources. If the link is down, please **contact me**, and I will restart it immediately.

![Docker Compose server](Images/architecture_live.png)

## Monitoring & Metrics

The project includes a comprehensive monitoring stack based on **Prometheus** and **Grafana**. It provides real-time insights into application performance, resource usage, and traffic patterns.

### Alerting Architecture
The system monitors the health of the application continuously.

1. **Prometheus** checks `up{job="nlp-app"}` every 5 seconds.
2. If the service is unreachable for **more than 1 minute**, an alert is fired.
3. **Alertmanager** receives the alert and pushes a notification to the configured **Telegram Chat**.

*(This ensures you sleep well, knowing the server will wake you up if it crashes!)*

### Live Access
You can view the raw metrics exposed by the application here:
**Metrics Endpoint:** [https://western-nlp.ddns.net/metrics](https://western-nlp.ddns.net/metrics)

![Docker Compose server](Images/Monitoring_and_Metrics.png)

### How to check locally
If you run the container locally, you can check metrics via curl:
```bash
curl http://localhost:8000/metrics
```
### Grafana Dashboard
Visualizes key metrics such as Requests Per Second (RPS), Latency (P99), Memory Usage, and HTTP Status Codes.

![Prometheus console](Images/prometheus_console.png)

![Grafana Dashboard](Images/grafana_dashboard.png)

## Load Testing & Capacity Planning

To determine the production limits of the current infrastructure (`AWS t3.micro`), we performed stress testing using **Locust**.

### Test Configuration
- **Tool:** Locust
- **Users:** 20 Concurrent Users
- **Spawn Rate:** 2 users/sec
- **Target:** `/sentiment` endpoint (DistilBERT model)

![Locust Setup](Images/locust_setup.png)

### Results Analysis
The test revealed the hardware limits of the single-core instance.

- **Stable Load:** Up to **5 RPS** (Requests Per Second) with acceptable latency.
- **Failure Point:** At ~15 concurrent users, the CPU saturates (100%), leading to **504 Gateway Timeouts**.
- **Max Latency:** Spiked to 60s (Nginx timeout limit) under stress.

![Load Charts](Images/locust_charts.png)

*The graph clearly shows the "Cliff of Death" where response time (Purple) skyrockets and RPS (Green) collapses due to CPU throttling.*

![Load Statistics](Images/locust_stats.png)

## Model Versioning & Reproducibility

To ensure **deterministic behavior** in production and avoid "silent failures," the system **does not** pull the `latest` version of models from HuggingFace. 

Instead, we enforce strict version control by pinning specific **Git SHA Hashes** in the inference pipeline. This guarantees that the model running in production today is mathematically identical to the one tested during development.

**Pinned Revisions:**
- **Sentiment Model:** `distilbert-base-uncased...` @ `714eb0f` (Dec 2023 Stable)
- **Translation Model:** `Helsinki-NLP/opus-mt-en-fr` @ `dd7f654` (Feb 2024 Stable)

## MLOps: Experiment Tracking

The project is fully integrated with **Weights & Biases (W&B)** to track model performance in production. Unlike standard system monitoring (Prometheus), W&B focuses on the **quality of the ML model**.

It logs:
- **Inputs & Outputs:** What users are asking and how the model responds.
- **Confidence Scores:** Tracks how "sure" the model is about its predictions.
- **System Resources:** Correlates inference time with CPU/Memory usage.

![W&B Dashboard](Images/wandb_dashboard.png)


## License

**MIT License**

Copyright (c) 2025 Andriy Vlonha

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
