ZentraX AI
A privacy-first, full-stack AI platform.

ZentraX AI pairs a modern Next.js frontend with an async Python (FastAPI) backend to deliver AI-powered conversations and tooling — without compromising on user privacy, security, or performance.

Table of Contents
Overview
Key Features
Tech Stack
Project Structure
Getting Started with Docker (Recommended)
Local Development Setup
Environment Variables
Contributing
License
Overview
ZentraX AI is built on a simple premise: powerful AI tooling shouldn't come at the cost of user privacy. Every service in the backend — chat, auth, and the extensible tool system — is designed around data minimization, secure-by-default practices, and clear boundaries between components.

The platform ships as two independently deployable services (frontend + backend), orchestrated together via Docker Compose behind an Nginx reverse proxy for production.

Key Features
🔒 Privacy-First by Design — opaque user references instead of raw PII in logs, metadata-only logging (no message content ever logged), configurable session/data retention, and GDPR-style account erasure built into the user service.
🎨 Modern Dark/Gold UI — a distinctive, polished interface built with Next.js, designed for clarity and focus during long AI sessions.
🧩 Modular Architecture — the backend is organized into clean, independently testable services (chat_service, user_service, toolkit_service) connected through explicit interfaces rather than tight coupling.
⚙️ Extensible Tool System — a pluggable registry lets new AI-callable tools and integrations be added without touching core conversation logic, complete with permissioning, timeouts, and schema validation.
🔐 Secure Authentication — Argon2id password hashing, short-lived JWT access tokens with refresh rotation, and timing-safe login to prevent account enumeration.
⚡ Fully Asynchronous — built end-to-end on async/await, from API routes down to database access and outbound provider calls.
🐳 Container-Native — production-ready Docker Compose setup with health checks, restart policies, and an Nginx reverse proxy with TLS termination.
🛡️ Sentinel Core — centralized runtime monitoring and safeguard logic in backend/app/core/sentinel_core.py.
✅ Compliance Guard — policy and compliance enforcement layer in backend/app/core/compliance_guard.py.
🧬 Feature Generator — automated feature derivation/generation utilities in backend/app/core/feature_generator.py.
🤝 External Agent Integration — a dedicated service for coordinating with external agents in backend/app/services/external_agent.py.
Tech Stack
Layer	Technology
Frontend	Next.js (React), TypeScript
Backend	Python, FastAPI, async/await
Database	PostgreSQL via SQLAlchemy 2.0 (async)
Auth	Argon2id, JWT (python-jose)
Reverse Proxy	Nginx (TLS termination, routing, rate limiting)
Containerization	Docker, Docker Compose
Caching / Sessions	Redis
Project Structure
zentrax-ai/
├── frontend/                      # Next.js application
│   ├── app/                       # App router pages & layouts
│   ├── components/                # Reusable UI components
│   ├── public/                    # Static assets
│   ├── Dockerfile
│   └── package.json
│
├── backend/                       # FastAPI application
│   ├── app/
│   │   ├── api/                   # Route definitions / endpoints
│   │   ├── services/               # Core business logic
│   │   │   ├── chat_service.py     # Conversation orchestration & AI provider dispatch
│   │   │   ├── user_service.py     # Registration, auth, profile management
│   │   │   ├── toolkit_service.py  # Modular AI tool registry & execution
│   │   │   └── external_agent.py   # Coordination with external agents
│   │   ├── repositories/          # Database access layer (SQLAlchemy)
│   │   ├── models/                 # SQLAlchemy models / Pydantic schemas
│   │   └── core/                   # Config, security helpers, dependencies
│   │       ├── sentinel_core.py     # Runtime monitoring & safeguard logic
│   │       ├── compliance_guard.py  # Policy & compliance enforcement
│   │       └── feature_generator.py # Automated feature derivation/generation
│   ├── requirements.txt
│   └── Dockerfile
│
├── infrastructure/
│   ├── docker-compose.yml          # Orchestrates frontend + backend
│   ├── .env.example                # Documents required environment variables
│   └── deployment/
│       ├── nginx.conf              # Reverse proxy: TLS, routing, security headers
│       └── deploy.sh               # Build → health-check → rollback deploy script
│
├── .gitignore
└── README.md
Getting Started with Docker (Recommended)
The fastest way to run the full stack is with Docker Compose — no local Node.js or Python setup required.

Prerequisites
Docker (20.10+)
Docker Compose (v2, bundled with modern Docker installs)
1. Clone the repository
git clone https://github.com/your-org/zentrax-ai.git
cd zentrax-ai
2. Configure environment variables
Copy the example env file and fill in real values (see Environment Variables below):

cp infrastructure/.env.example infrastructure/.env
⚠️ infrastructure/.env contains secrets — it's git-ignored by default. Never commit real credentials.

3. Build and start the stack
cd infrastructure
docker compose up --build
This will:

Build the frontend image and serve it on port 3000
Build the backend image and serve it on port 8000
Wire both services together on an isolated Docker network
Apply restart policies so containers recover automatically from crashes
4. Verify it's running
Frontend: http://localhost:3000
Backend health check: http://localhost:8000/health
Backend API docs (Swagger UI): http://localhost:8000/docs
5. Stop the stack
docker compose down
Production deployment
For a production rollout behind the included Nginx reverse proxy (TLS termination, security headers, rate limiting), see infrastructure/deployment/nginx.conf and run the deploy script:

cd infrastructure/deployment
./deploy.sh
The script builds fresh images, brings the stack up, waits for both services to pass health checks, and automatically rolls back if they don't.

Local Development Setup
For active development, running the frontend and backend natively (outside Docker) gives you faster reload cycles.

Backend (FastAPI)
Prerequisites: Python 3.12+

cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp ../infrastructure/.env.example .env

# Run database migrations (if using Alembic)
alembic upgrade head

# Start the dev server with hot reload
uvicorn app.main:app --reload --port 8000
The API will be available at http://localhost:8000, with interactive docs at http://localhost:8000/docs.

Frontend (Next.js)
Prerequisites: Node.js 20+

cd frontend

# Install dependencies
npm install

# Copy and configure environment variables
cp .env.example .env.local

# Start the dev server
npm run dev
The app will be available at http://localhost:3000, automatically proxying API requests to your local backend.

Running both together
Open two terminal sessions — one for backend/ (uvicorn ... --reload) and one for frontend/ (npm run dev) — and both will hot-reload independently as you work.

Environment Variables
Key variables required by the backend (see infrastructure/.env.example for the full list):

Variable	Description
DATABASE_URL	PostgreSQL connection string
JWT_SECRET	Secret used to sign auth tokens (min. 32 characters, keep private)
NEXT_PUBLIC_API_URL	Public URL the frontend uses to reach the backend
IMAGE_TAG	Docker image tag used by Compose (defaults to latest)
All secrets should be sourced from a secrets manager (AWS Secrets Manager, Vault, etc.) in real production deployments — .env files are for local/dev convenience only.

Contributing
Contributions are welcome. Please open an issue to discuss significant changes before submitting a pull request, and ensure new backend services follow the existing pattern of dependency-injected interfaces (e.g. SessionStore, UserRepository, ToolRegistry) so components stay testable and swappable.

License
Add your chosen license here (e.g. MIT, Apache 2.0) — none is specified yet.
