# NLP Inference Microservice (Docker Compose & Redis)

![CI Pipeline](https://github.com/Western-1/nlp-inference-service/actions/workflows/ci.yml/badge.svg) 
![Python Version](https://img.shields.io/badge/python-3.9-blue.svg) 
![License](https://img.shields.io/badge/license-MIT-green.svg) 
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white) 
![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?style=flat&logo=redis&logoColor=white) 
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)

A production-ready **Microservices Architecture** for Natural Language Processing. This project orchestrates multiple containers using **Docker Compose**: a FastAPI application for inference and a **Redis** database for high-speed logging and persistence.

It features a fully automated **CI/CD Pipeline** via GitHub Actions.

![Dashboard Overview](Images/1.png)

## Architecture & Workflow

This project demonstrates a modern microservices approach. Instead of a monolithic script, the system decouples inference from data persistence and includes automated testing pipelines.

```mermaid
graph LR
  %% --- Styling Definitions ---
  classDef app fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:#000
  classDef db fill:#ffcdd2,stroke:#c62828,stroke-width:2px,color:#000
  classDef ext fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
  classDef planned fill:#fafafa,stroke:#9e9e9e,stroke-width:2px,stroke-dasharray: 5 5,color:#616161
  classDef net fill:none,stroke:#546e7a,stroke-width:2px,stroke-dasharray: 5 5,color:#546e7a

  %% --- Actors ---
  User([User / Client])
  style User fill:#fff,stroke:#333,stroke-width:2px,color:#000

  %% --- Ingress Layer (Planned) ---
  subgraph Ingress [Ingress Layer]
    direction TB
    Nginx[Nginx Reverse Proxy]:::planned
  end

  %% --- Private Docker Network ---
  subgraph DockerNet [Private Docker Network]
    
    %% App Service
    subgraph Container_App [App Service]
      direction TB
      Gunicorn[Gunicorn Manager]:::app
      subgraph Workers [Async Workers]
        Uvicorn[Uvicorn Worker]:::app
        Logic[Lazy Load Logic]:::app
      end
    end

    %% Storage Service
    Redis[(Redis DB)]:::db
  end

  %% --- External / Cloud ---
  subgraph External [External Services]
    HF_Hub[HuggingFace Hub]:::ext
    HFCache[Volume: HF Cache]:::ext
    S3[(S3 Archive)]:::planned
    Prometheus[Metrics]:::planned
  end

  %% === Data Flow ===
  User -->|"1. HTTPS Request"| Nginx
  Nginx -.->|"2. Forward (Planned)"| Gunicorn
  User -.->|"Direct Access (Dev)"| Gunicorn

  Gunicorn -->|"3. Spawn Processes"| Uvicorn
  Uvicorn -->|"4. Inference"| Logic
  
  Logic -.->|"Download (First run)"| HF_Hub
  Logic -->|"Load from"| HFCache
  
  Uvicorn -->|"5. LPUSH (Async)"| Redis
  Redis -->|"LTRIM (Auto-cleanup)"| Redis
  
  User -->|"GET /history"| Uvicorn
  Uvicorn <-->|"LRANGE"| Redis

  Redis -.->|"Archive (Planned)"| S3
  Uvicorn -.->|"/metrics (Planned)"| Prometheus

  style DockerNet fill:none,stroke:#607d8b,stroke-width:2px,stroke-dasharray: 5 5
  style Ingress fill:none,stroke:none
  style External fill:none,stroke:none
  style Container_App fill:#f1f8e9,stroke:#558b2f,stroke-width:1px
  style Workers fill:#fff,stroke:none
```

## Key Features

- **Microservices Orchestration:** Fully dockerized environment via `docker-compose`.
- **CI/CD Pipeline:** Automated testing via GitHub Actions on every push.
- **Multi-Model Inference:** `DistilBERT` (Sentiment) & `Helsinki-NLP` (Translation).
- **Persistent Storage:** Asynchronous logging to Redis using `LPUSH`/`LTRIM`.
- **Mocked Testing:** Unit tests use `unittest.mock` to simulate ML models and Redis in CI environments.

## Tech Stack

- **Orchestration:** Docker Compose
- **CI/CD:** GitHub Actions
- **Core:** Python 3.9, FastAPI, Uvicorn
- **Database:** Redis (Alpine)
- **ML Backend:** PyTorch, Transformers
- **Models:**
  - `distilbert-base-uncased-finetuned-sst-2-english`
  - `Helsinki-NLP/opus-mt-en-fr`

## Project Structure

```
.
├── .github/workflows/   # CI/CD Pipeline configuration
├── tests/               # Unit & Integration tests
├── Images/              # Documentation assets
├── docker-compose.yml   # Service orchestration (App + Redis)
├── Dockerfile           # App container configuration
├── main.py              # Application logic & endpoints
└── requirements.txt     # Python dependencies
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

## Development & Testing

This project uses **Pytest** for unit and integration testing. The CI pipeline runs these tests automatically.

To run tests locally:
```bash
pip install pytest httpx

pytest tests/ -v
```

## Deployment (AWS EC2)

This service is designed to be deployed on AWS EC2 (Ubuntu Server).

1. **Provision Infrastructure:**
   - Launch `t3.micro` instance (Ubuntu 24.04).
   - Configure Security Group: Open ports `22` (SSH) and `8000` (API).
   - Configure Swap file (4GB) to handle ML model memory requirements.

2. **Deploy:**
```bash
ssh -i key.pem ubuntu@<EC2_IP>
git clone https://github.com/Western-1/nlp-inference-service
docker compose up -d --build
```

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
**[http://western-nlp.ddns.net/docs](http://western-nlp.ddns.net/docs)**

> [!IMPORTANT]
> Active (Hosted on AWS EC2 Free Tier)

> [!NOTE]
> Since this is a free tier instance, it might be stopped to save resources. If the link is down, please **contact me**, and I will restart it immediately.

## License

**MIT License**

Copyright (c) 2025 Andriy Vlonha

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
