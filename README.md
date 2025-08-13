# Mini Project 1 : CCTV Face Recognition System with Vector Embeddings

## Complete Technical Documentation


### Last Updated: August 14, 2025

### Development Team: [Saurav Kale](https://github.com/ksaurav24), [Mithilesh Kolhapurkar](https://github.com/MITHILESHK11), [Ramjan Shaikh](https://github.com/demonxg)


***

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technical Features](#3-technical-features)
4. [Technology Stack](#4-technology-stack)
5. [Project Structure](#5-project-structure)
6. [Installation \& Setup](#6-installation--setup)
7. [Development Workflow](#7-development-workflow)
8. [API Documentation](#8-api-documentation)
9. [Database Schema](#9-database-schema)
10. [Security \& Privacy](#10-security--privacy)
11. [Performance Specifications](#11-performance-specifications)
12. [Testing Strategy](#12-testing-strategy)
13. [Deployment Guide](#13-deployment-guide)

***

## 1. Project Overview

### 1.1 System Description

The **CCTV Face Recognition System** is a privacy-preserving surveillance solution designed for residential societies and commercial buildings. The system uses advanced computer vision and machine learning to identify authorized members while tracking unknown individuals through mathematical embeddings rather than storing actual biometric images.

### 1.2 Key Innovation: Hybrid Privacy Approach

- **Known Members**: Only mathematical embeddings (128-512 dimensions) stored - **zero face images**
- **Unknown Users**: Temporary encrypted face images for admin identification with auto-deletion
- **Vector Database**: Optimized for real-time similarity search across thousands of members


### 1.3 Core Objectives

```mermaid
graph TD
    A[CCTV Camera Feed] --> B[Face Detection]
    B --> C[Feature Extraction]
    C --> D[Vector Database Query]
    D --> E{Known Member?}
    E -->|Yes| F[Log Entry & Allow]
    E -->|No| G[Unknown User Handler]
    G --> H[Store Encrypted Image]
    G --> I[Generate Unique ID]
    G --> J[Admin Notification]
    J --> K[Admin Identifies User]
    K --> L[Convert to Member]
```


### 1.4 Business Use Cases

- **Residential Societies**: Automated gate access, visitor tracking
- **Corporate Offices**: Employee attendance, unauthorized access prevention
- **Retail Stores**: VIP customer recognition, shoplifting prevention
- **Educational Institutions**: Student safety, visitor management

***

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CCTV Layer    │    │  Processing     │    │   Storage       │
│                 │    │     Layer       │    │    Layer        │
│ • IP Cameras    │────│ • Face Detect   │────│ • Vector DB     │
│ • Video Streams │    │ • Feature Ext   │    │ • PostgreSQL    │
│ • Edge Process  │    │ • Recognition   │    │ • Redis Cache   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Application    │
                    │     Layer       │
                    │ • REST API      │
                    │ • Admin Portal  │
                    │ • Alert System  │
                    └─────────────────┘
```


### 2.2 Component Overview

#### 2.2.1 Face Recognition Pipeline

```python
# Core recognition flow
def recognition_pipeline(frame):
    faces = face_detector.detect(frame)           # MTCNN/Haar
    for face in faces:
        aligned = face_aligner.align(face)        # Geometric normalization  
        embedding = feature_extractor.extract(aligned)  # FaceNet/ArcFace
        matches = vector_db.similarity_search(embedding)  # Cosine similarity
        return classify_detection(matches)
```


#### 2.2.2 Vector Database Integration

- **Primary**: Pinecone/Qdrant for production scalability
- **Development**: Chromadb for local testing
- **Indexing**: HNSW (Hierarchical Navigable Small World) algorithm
- **Metrics**: Euclidean/Cosine distance for face matching


### 2.3 Data Flow Architecture

```
Input Stream → Face Detection → Preprocessing → Feature Extraction 
     ↓
Vector Query → Similarity Scoring → Classification → Action Trigger
     ↓
Database Update → Alert Generation → Admin Dashboard → Response
```


***

## 3. Technical Features

### 3.1 Core Features

#### 3.1.1 Member Management System

```python
class MemberManager:
    def register_member(self, member_data, photos):
        """
        Complete member registration with embedding generation
        - Multiple photo processing (3-5 angles)
        - Quality validation (lighting, resolution, pose)
        - Embedding extraction and storage
        - Database record creation
        """
        
    def generate_embeddings(self, photos):
        """
        Process multiple photos to create robust member profile
        - Face detection and validation
        - Geometric alignment and normalization
        - Feature extraction using pre-trained CNN
        - Centroid embedding calculation
        """
```


#### 3.1.2 Unknown User Tracking

```python
class UnknownUserHandler:
    def detect_unknown(self, embedding, face_image):
        """
        Handle unknown person detection
        - Check existing unknown users (similarity search)
        - Generate unique unknown ID if new
        - Store encrypted face image temporarily
        - Create tracking record with metadata
        """
        
    def update_unknown_appearance(self, unknown_id, new_embedding):
        """
        Track recurring unknown users
        - Update appearance count
        - Store additional embeddings for accuracy
        - Log camera locations and timestamps
        - Trigger admin alerts for frequent visitors
        """
```


#### 3.1.3 Real-time Recognition Engine

```python
class RecognitionEngine:
    def process_cctv_frame(self, frame, camera_id):
        """
        Real-time face recognition from CCTV feed
        - Multi-face detection in single frame
        - Parallel embedding extraction
        - Batch vector database queries
        - Confidence-based classification
        """
        
    def classify_detection(self, similarity_scores):
        """
        Classification logic with configurable thresholds
        - Known member: >0.85 similarity
        - Recurring unknown: 0.80-0.85 similarity
        - New unknown: <0.80 similarity
        """
```


### 3.2 Advanced Features

#### 3.2.1 Privacy-Preserving Architecture

- **Embedding Storage**: Only mathematical representations, no raw images
- **Temporary Image Storage**: Encrypted unknown user photos with auto-deletion
- **GDPR Compliance**: Right to erasure, data minimization principles
- **Audit Logging**: Complete activity tracking for compliance


#### 3.2.2 Intelligent Alert System

```python
class AlertSystem:
    def __init__(self):
        self.alert_rules = {
            'new_unknown': {'priority': 'high', 'immediate': True},
            'recurring_unknown': {'priority': 'medium', 'threshold': 5},
            'unauthorized_access': {'priority': 'critical', 'immediate': True},
            'system_error': {'priority': 'high', 'technical': True}
        }
    
    def process_alert(self, alert_type, data):
        """
        Intelligent alert processing with rule-based routing
        - Email notifications for critical events
        - Dashboard updates for all events  
        - SMS alerts for security breaches
        - Slack/Teams integration for team notifications
        """
```


#### 3.2.3 Multi-Camera Support

- **Distributed Processing**: Edge computing capabilities
- **Camera Synchronization**: Timestamp alignment across feeds
- **Load Balancing**: Intelligent frame processing distribution
- **Failover Support**: Automatic camera switching on failure

***

## 4. Technology Stack

### 4.1 Core Technologies

#### 4.1.1 Machine Learning \& Computer Vision

```yaml
Deep Learning Frameworks:
  - TensorFlow: 2.13.0+ (Primary ML framework)
  - PyTorch: 1.13.0+ (Alternative for custom models)
  - OpenCV: 4.8.0+ (Computer vision operations)

Pre-trained Models:
  - FaceNet: 128-dim embeddings, proven accuracy
  - MTCNN: Multi-task face detection and alignment  
  - ArcFace: 512-dim embeddings, state-of-the-art
  - RetinaFace: Advanced face detection for challenging conditions
```


#### 4.1.2 Vector Databases

```yaml
Production Options:
  - Pinecone: Cloud-managed, excellent scaling
  - Qdrant: Open-source, Rust-based performance
  - Weaviate: GraphQL interface, hybrid search

Development:
  - ChromaDB: Local development and testing
  - FAISS: Facebook's similarity search library
```


#### 4.1.3 Backend Infrastructure

```yaml
Web Framework:
  - FastAPI: 0.104.0+ (High-performance async API)
  - Alternative: Flask 2.3.0+ (Traditional WSGI)

Database Systems:
  - PostgreSQL: 15.0+ (Primary relational database)
  - Redis: 7.0+ (Caching and session storage)
  - SQLite: Development and testing

Message Queue:
  - Celery: Distributed task processing
  - Redis/RabbitMQ: Message broker options
```


#### 4.1.4 Frontend Technologies

```yaml
Web Interface:
  - React: 18.0+ with TypeScript
  - Material-UI: Component library
  - Chart.js: Analytics visualization

Mobile (Future):
  - React Native: Cross-platform support
  - Flutter: Alternative mobile framework
```


### 4.2 Infrastructure \& DevOps

```yaml
Containerization:
  - Docker: 24.0+ (Application containerization)
  - Docker Compose: Multi-service development

Orchestration:
  - Kubernetes: Production deployment
  - Helm: Package management for K8s

Monitoring:
  - Prometheus: Metrics collection
  - Grafana: Visualization dashboards
  - ELK Stack: Logging and search

Security:
  - HTTPS/TLS: Encrypted communications
  - JWT: Authentication tokens
  - OAuth2: Third-party authentication
```


### 4.3 Hardware Requirements

#### 4.3.1 Minimum Specifications

```yaml
Processing Server:
  - CPU: Intel i7-9700K / AMD Ryzen 7 3700X
  - RAM: 16GB DDR4
  - GPU: NVIDIA GTX 1660 Ti / RTX 3060 (4GB VRAM)
  - Storage: 500GB NVMe SSD
  - Network: Gigabit Ethernet

Camera Requirements:
  - Resolution: Minimum 720p (1080p preferred)
  - Frame Rate: 15-30 FPS
  - Protocol: RTSP/HTTP streaming
  - Night Vision: IR illumination support
  - Lens: Fixed focal length, minimal distortion
```


#### 4.3.2 Recommended Production Setup

```yaml
Processing Cluster:
  - CPU: Intel Xeon Gold / AMD EPYC (16+ cores)
  - RAM: 64GB+ DDR4 ECC
  - GPU: NVIDIA RTX 4080/A4000 (16GB+ VRAM)
  - Storage: 2TB+ NVMe RAID 1
  - Network: 10GbE with redundancy

Load Balancer:
  - HAProxy/NGINX: Traffic distribution
  - SSL Termination: Certificate management
  - Health Checks: Automatic failover
```


***

## 5. Project Structure

### 5.1 Directory Layout

```
face_recognition_cctv/
├── README.md                          # This documentation
├── docker-compose.yml                 # Development environment
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
├── pyproject.toml                     # Project configuration
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── main.py                       # Application entry point
│   ├── config/                       # Configuration management
│   │   ├── __init__.py
│   │   ├── settings.py               # Application settings
│   │   ├── database.py               # Database configuration
│   │   └── logging.py                # Logging configuration
│   │
│   ├── core/                         # Core business logic
│   │   ├── __init__.py
│   │   ├── face_detection/           # Face detection modules
│   │   │   ├── __init__.py
│   │   │   ├── mtcnn_detector.py     # MTCNN implementation
│   │   │   ├── haar_detector.py      # Haar cascade detector
│   │   │   └── detector_factory.py   # Detector selection
│   │   │
│   │   ├── face_recognition/         # Recognition modules
│   │   │   ├── __init__.py
│   │   │   ├── facenet_extractor.py  # FaceNet embeddings
│   │   │   ├── arcface_extractor.py  # ArcFace embeddings
│   │   │   ├── face_aligner.py       # Geometric alignment
│   │   │   └── embedding_utils.py    # Vector operations
│   │   │
│   │   ├── vector_database/          # Vector DB integration
│   │   │   ├── __init__.py
│   │   │   ├── pinecone_client.py    # Pinecone integration
│   │   │   ├── qdrant_client.py      # Qdrant integration
│   │   │   ├── base_client.py        # Abstract base class
│   │   │   └── similarity_search.py  # Search algorithms
│   │   │
│   │   └── recognition_engine/       # Main recognition engine
│   │       ├── __init__.py
│   │       ├── pipeline.py           # Recognition pipeline
│   │       ├── member_manager.py     # Member operations
│   │       ├── unknown_handler.py    # Unknown user handling
│   │       └── real_time_processor.py # CCTV processing
│   │
│   ├── api/                          # REST API endpoints
│   │   ├── __init__.py
│   │   ├── app.py                    # FastAPI application
│   │   ├── dependencies.py           # Dependency injection
│   │   ├── middleware.py             # Custom middleware
│   │   │
│   │   ├── routes/                   # API route definitions
│   │   │   ├── __init__.py
│   │   │   ├── auth.py               # Authentication routes
│   │   │   ├── members.py            # Member management
│   │   │   ├── unknown_users.py      # Unknown user management
│   │   │   ├── recognition.py        # Real-time recognition
│   │   │   ├── cameras.py            # Camera management
│   │   │   └── admin.py              # Admin operations
│   │   │
│   │   ├── schemas/                  # Pydantic models
│   │   │   ├── __init__.py
│   │   │   ├── member.py             # Member data models
│   │   │   ├── unknown_user.py       # Unknown user models
│   │   │   ├── recognition.py        # Recognition response models
│   │   │   └── common.py             # Shared models
│   │   │
│   │   └── security/                 # Security utilities
│   │       ├── __init__.py
│   │       ├── authentication.py     # JWT handling
│   │       ├── authorization.py      # Permission management
│   │       └── encryption.py         # Data encryption
│   │
│   ├── database/                     # Database layer
│   │   ├── __init__.py
│   │   ├── connection.py             # Database connections
│   │   ├── models/                   # SQLAlchemy models
│   │   │   ├── __init__.py
│   │   │   ├── member.py             # Member table model
│   │   │   ├── unknown_user.py       # Unknown user model
│   │   │   ├── appearance_log.py     # Appearance tracking
│   │   │   ├── camera.py             # Camera configuration
│   │   │   └── alert.py              # Alert records
│   │   │
│   │   ├── repositories/             # Data access layer
│   │   │   ├── __init__.py
│   │   │   ├── member_repository.py
│   │   │   ├── unknown_repository.py
│   │   │   └── base_repository.py
│   │   │
│   │   └── migrations/               # Database migrations
│   │       ├── versions/
│   │       └── alembic.ini
│   │
│   ├── services/                     # Business services
│   │   ├── __init__.py
│   │   ├── member_service.py         # Member business logic
│   │   ├── recognition_service.py    # Recognition orchestration
│   │   ├── alert_service.py          # Alert management
│   │   ├── camera_service.py         # Camera integration
│   │   └── encryption_service.py     # Image encryption
│   │
│   ├── utils/                        # Utility functions
│   │   ├── __init__.py
│   │   ├── image_utils.py            # Image processing helpers
│   │   ├── vector_utils.py           # Vector operations
│   │   ├── validation.py             # Input validation
│   │   └── logger.py                 # Logging utilities
│   │
│   └── workers/                      # Background tasks
│       ├── __init__.py
│       ├── celery_app.py             # Celery configuration
│       ├── recognition_worker.py     # Background recognition
│       ├── cleanup_worker.py         # Image cleanup tasks
│       └── notification_worker.py    # Alert notifications
│
├── frontend/                         # Web interface
│   ├── public/
│   ├── src/
│   │   ├── components/               # React components
│   │   │   ├── admin/                # Admin dashboard
│   │   │   ├── member/               # Member management
│   │   │   ├── unknown/              # Unknown user management
│   │   │   └── common/               # Shared components
│   │   │
│   │   ├── pages/                    # Page components
│   │   ├── hooks/                    # Custom React hooks
│   │   ├── services/                 # API client services
│   │   └── utils/                    # Frontend utilities
│   │
│   ├── package.json
│   └── tsconfig.json
│
├── data/                             # Data storage
│   ├── models/                       # Pre-trained models
│   │   ├── facenet/                  # FaceNet model files
│   │   ├── mtcnn/                    # MTCNN model files
│   │   └── arcface/                  # ArcFace model files
│   │
│   ├── unknown_faces_temp/           # Temporary encrypted images
│   ├── member_photos/                # Member registration photos
│   └── logs/                         # Application logs
│
├── scripts/                          # Utility scripts
│   ├── setup_database.py             # Database initialization
│   ├── setup_vector_db.py            # Vector database setup
│   ├── model_downloader.py           # Download pre-trained models
│   ├── data_migration.py             # Data migration utilities
│   └── performance_test.py           # Performance benchmarking
│
├── tests/                            # Test suite
│   ├── unit/                         # Unit tests
│   │   ├── test_face_detection.py
│   │   ├── test_recognition.py
│   │   ├── test_vector_db.py
│   │   └── test_unknown_handler.py
│   │
│   ├── integration/                  # Integration tests
│   │   ├── test_api_endpoints.py
│   │   ├── test_recognition_pipeline.py
│   │   └── test_database_operations.py
│   │
│   ├── load/                         # Load testing
│   │   ├── test_concurrent_recognition.py
│   │   └── test_camera_throughput.py
│   │
│   ├── fixtures/                     # Test data
│   │   ├── sample_faces/
│   │   └── mock_data.json
│   │
│   └── conftest.py                   # Pytest configuration
│
├── docs/                             # Documentation
│   ├── api/                          # API documentation
│   ├── deployment/                   # Deployment guides
│   ├── architecture/                 # System architecture
│   └── user_manual/                  # User documentation
│
├── deployment/                       # Deployment configurations
│   ├── docker/                       # Docker configurations
│   │   ├── Dockerfile.api            # API service container
│   │   ├── Dockerfile.worker         # Worker service container
│   │   └── Dockerfile.frontend       # Frontend container
│   │
│   ├── kubernetes/                   # K8s manifests
│   │   ├── namespace.yaml
│   │   ├── deployments/
│   │   ├── services/
│   │   └── ingress.yaml
│   │
│   └── ansible/                      # Infrastructure automation
│       ├── playbooks/
│       └── roles/
│
└── monitoring/                       # Monitoring configuration
    ├── prometheus/                   # Metrics configuration
    ├── grafana/                      # Dashboard definitions
    └── alerts/                       # Alert rules
```


### 5.2 Key Configuration Files

#### 5.2.1 Environment Configuration (.env)

```bash
# Application Configuration
APP_NAME=CCTV Face Recognition System
APP_VERSION=1.0.0
DEBUG=False
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/face_recognition
REDIS_URL=redis://localhost:6379/0

# Vector Database Configuration
VECTOR_DB_TYPE=pinecone  # pinecone, qdrant, chromadb
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Security Configuration
SECRET_KEY=your_secret_key_here
ENCRYPTION_KEY=your_encryption_key_here
JWT_EXPIRATION_HOURS=24

# Recognition Configuration
FACE_DETECTION_MODEL=mtcnn  # mtcnn, haar, retinaface
FACE_RECOGNITION_MODEL=facenet  # facenet, arcface
RECOGNITION_THRESHOLD=0.85
UNKNOWN_THRESHOLD=0.80

# Camera Configuration
MAX_CAMERAS=10
RTSP_TIMEOUT=30
FRAME_PROCESSING_INTERVAL=3

# Storage Configuration
UNKNOWN_IMAGE_RETENTION_DAYS=30
MAX_EMBEDDINGS_PER_USER=5
IMAGE_ENCRYPTION_ENABLED=True

# Alert Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your_email@example.com
EMAIL_PASSWORD=your_email_password
ADMIN_EMAIL=admin@company.com

# Performance Configuration
MAX_WORKERS=4
BATCH_SIZE=32
CACHE_TTL=3600
```


#### 5.2.2 Docker Compose (docker-compose.yml)

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: deployment/docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/face_recognition
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
      - vector_db
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs

  worker:
    build:
      context: .
      dockerfile: deployment/docker/Dockerfile.worker
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/face_recognition
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./data:/app/data

  frontend:
    build:
      context: ./frontend
      dockerfile: ../deployment/docker/Dockerfile.frontend
    ports:
      - "3000:3000"
    depends_on:
      - api

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=face_recognition
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  vector_db:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
```


***

## 6. Installation \& Setup

### 6.1 Prerequisites

#### 6.1.1 System Requirements

```bash
# Ubuntu 20.04+ / CentOS 8+ / macOS 12+
# Python 3.9+
# Docker 20.10+
# Git 2.30+

# GPU Support (Optional but Recommended)
# NVIDIA GPU with CUDA 11.8+
# NVIDIA Driver 470+
```


#### 6.1.2 Development Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/your-org/face-recognition-cctv.git
cd face-recognition-cctv

# 2. Create Python virtual environment
python3.9 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# 3. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install development dependencies
pip install -r requirements-dev.txt

# 5. Set up pre-commit hooks
pre-commit install

# 6. Copy environment configuration
cp .env.example .env
# Edit .env with your configuration
```


#### 6.1.3 Model Downloads

```bash
# Download pre-trained models
python scripts/model_downloader.py --all

# Or download specific models
python scripts/model_downloader.py --model facenet
python scripts/model_downloader.py --model mtcnn
```


### 6.2 Database Setup

#### 6.2.1 PostgreSQL Database

```bash
# Start PostgreSQL with Docker
docker run --name postgres-face-recognition \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=face_recognition \
  -p 5432:5432 -d postgres:15-alpine

# Initialize database schema
python scripts/setup_database.py

# Run database migrations
alembic upgrade head
```


#### 6.2.2 Vector Database Setup

```bash
# For Qdrant (Local development)
docker run --name qdrant-face-recognition \
  -p 6333:6333 -d qdrant/qdrant

# Initialize vector database
python scripts/setup_vector_db.py --provider qdrant

# For Pinecone (Production)
# Set PINECONE_API_KEY in .env file
python scripts/setup_vector_db.py --provider pinecone
```


### 6.3 Quick Start

#### 6.3.1 Development Server

```bash
# Start all services with Docker Compose
docker-compose up -d

# Or start individual components:

# 1. Start API server
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# 2. Start background workers
celery -A src.workers.celery_app worker --loglevel=info

# 3. Start frontend (separate terminal)
cd frontend
npm install
npm start

# Access the application:
# Frontend: http://localhost:3000
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```


#### 6.3.2 Initial Configuration

```bash
# Create admin user
python scripts/create_admin_user.py \
  --username admin \
  --email admin@company.com \
  --password secure_password

# Add test cameras
python scripts/add_test_cameras.py

# Register test members
python scripts/register_test_members.py
```


***

## 7. Development Workflow

### 7.1 Code Organization Principles

#### 7.1.1 Separation of Concerns

```python
# Clear separation between layers:
# 1. API Layer (routes, schemas, validation)
# 2. Service Layer (business logic)
# 3. Repository Layer (data access)
# 4. Core Layer (domain models, algorithms)

# Example: Member registration flow
# Route -> Service -> Repository -> Database
# Route -> Service -> Core Engine -> Vector DB
```


#### 7.1.2 Dependency Injection

```python
# Use FastAPI's dependency injection system
from fastapi import Depends
from src.services.member_service import MemberService
from src.core.recognition_engine import RecognitionEngine

async def get_member_service() -> MemberService:
    return MemberService()

@app.post("/members/register")
async def register_member(
    member_data: MemberCreateSchema,
    service: MemberService = Depends(get_member_service)
):
    return await service.register_member(member_data)
```


### 7.2 Development Standards

#### 7.2.1 Code Style

```bash
# Use Black for code formatting
black src/ tests/

# Use isort for import sorting
isort src/ tests/

# Use flake8 for linting
flake8 src/ tests/

# Use mypy for type checking
mypy src/
```


#### 7.2.2 Testing Standards

```python
# Write comprehensive tests for all components

# Unit Tests: Individual functions/classes
def test_face_detection():
    detector = MTCNNDetector()
    faces = detector.detect(sample_image)
    assert len(faces) == 1
    assert faces[^0]['confidence'] > 0.9

# Integration Tests: Component interactions
@pytest.mark.asyncio
async def test_member_registration_flow():
    member_data = create_test_member()
    result = await member_service.register_member(member_data)
    assert result['success'] is True
    
    # Verify in database
    stored_member = await member_repository.get_by_id(result['member_id'])
    assert stored_member is not None

# Load Tests: Performance validation
def test_concurrent_recognition():
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(recognition_engine.process_frame, frame)
            for frame in test_frames
        ]
        results = [f.result() for f in futures]
    
    assert all(r['success'] for r in results)
```


#### 7.2.3 Error Handling

```python
# Use custom exceptions for different error types
class FaceRecognitionError(Exception):
    """Base exception for face recognition errors"""
    pass

class NoFaceDetectedError(FaceRecognitionError):
    """Raised when no face is detected in image"""
    pass

class VectorDatabaseError(FaceRecognitionError):
    """Raised when vector database operations fail"""
    pass

# Handle errors gracefully with proper logging
import logging

logger = logging.getLogger(__name__)

def process_recognition(image):
    try:
        faces = detector.detect(image)
        if not faces:
            raise NoFaceDetectedError("No face detected in image")
        
        return extract_embeddings(faces)
        
    except NoFaceDetectedError:
        logger.warning("No face detected in provided image")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in recognition: {e}")
        raise FaceRecognitionError(f"Recognition failed: {e}")
```


### 7.3 Git Workflow

#### 7.3.1 Branch Strategy

```bash
# Main branches:
# - main: Production-ready code
# - develop: Integration branch for features
# - feature/*: Individual feature development
# - hotfix/*: Critical bug fixes

# Feature development workflow:
git checkout develop
git pull origin develop
git checkout -b feature/unknown-user-tracking
# ... develop feature ...
git add .
git commit -m "feat: add unknown user tracking functionality"
git push origin feature/unknown-user-tracking
# Create pull request to develop branch
```


#### 7.3.2 Commit Message Convention

```bash
# Use conventional commits format:
# type(scope): description

# Types:
# feat: New feature
# fix: Bug fix
# docs: Documentation changes
# style: Code style changes
# refactor: Code refactoring
# test: Test additions/modifications
# chore: Build process or auxiliary tool changes

# Examples:
git commit -m "feat(recognition): add support for ArcFace embeddings"
git commit -m "fix(api): resolve member registration validation error"
git commit -m "docs(readme): update installation instructions"
```


***

## 8. API Documentation

### 8.1 Authentication

#### 8.1.1 JWT Authentication

```bash
# Login to get JWT token
POST /api/v1/auth/login
Content-Type: application/json

{
    "username": "admin@company.com",
    "password": "secure_password"
}

# Response:
{
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer",
    "expires_in": 86400
}

# Use token in subsequent requests:
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```


### 8.2 Core API Endpoints

#### 8.2.1 Member Management

```yaml
# Register New Member
POST /api/v1/members/register
Content-Type: multipart/form-data
Authorization: Bearer {token}

FormData:
  name: "John Doe"
  apartment: "A-101"
  email: "john.doe@email.com"
  phone: "+1234567890"
  photos: [file1.jpg, file2.jpg, file3.jpg]  # 3-5 photos

Response:
{
  "success": true,
  "member_id": "MEM_20250814_001",
  "embeddings_count": 3,
  "quality_score": 0.95,
  "message": "Member registered successfully"
}

# Get Member Details
GET /api/v1/members/{member_id}
Authorization: Bearer {token}

Response:
{
  "member_id": "MEM_20250814_001",
  "name": "John Doe",
  "apartment": "A-101",
  "status": "active",
  "registration_date": "2025-08-14T10:30:00Z",
  "last_seen": "2025-08-14T15:45:00Z",
  "total_appearances": 24
}

# List All Members
GET /api/v1/members?page=1&limit=20&search=john
Authorization: Bearer {token}

Response:
{
  "members": [...],
  "total_count": 150,
  "page": 1,
  "limit": 20,
  "total_pages": 8
}

# Update Member Information
PUT /api/v1/members/{member_id}
Content-Type: application/json
Authorization: Bearer {token}

{
  "name": "John Smith",
  "apartment": "A-102",
  "phone": "+1234567891"
}

# Delete Member
DELETE /api/v1/members/{member_id}
Authorization: Bearer {token}
```


#### 8.2.2 Real-time Recognition

```yaml
# Process Single Frame
POST /api/v1/recognition/process-frame
Content-Type: multipart/form-data
Authorization: Bearer {token}

FormData:
  frame: image_file.jpg
  camera_id: "camera_001"
  location: "Main Entrance"

Response:
{
  "timestamp": "2025-08-14T16:20:00Z",
  "camera_id": "camera_001",
  "detections": [
    {
      "type": "known_member",
      "member_id": "MEM_20250814_001",
      "name": "John Doe",
      "confidence": 0.92,
      "bbox": [100, 150, 200, 250]
    },
    {
      "type": "new_unknown",
      "unknown_id": "UNKNOWN_20250814_a1b2c3d4",
      "name": "Unknown Person a1b2c3d4",
      "bbox": [300, 200, 400, 300]
    }
  ]
}

# Get Recognition Statistics
GET /api/v1/recognition/stats?date_from=2025-08-01&date_to=2025-08-14
Authorization: Bearer {token}

Response:
{
  "total_detections": 1250,
  "known_members": 950,
  "unknown_users": 300,
  "accuracy": 0.96,
  "daily_stats": [
    {
      "date": "2025-08-14",
      "detections": 85,
      "known": 70,
      "unknown": 15
    }
  ]
}

# Start/Stop Camera Recognition
POST /api/v1/recognition/camera/{camera_id}/start
DELETE /api/v1/recognition/camera/{camera_id}/stop
Authorization: Bearer {token}
```


#### 8.2.3 Unknown User Management

```yaml
# Get Unknown Users List
GET /api/v1/unknown-users?status=unidentified&limit=20
Authorization: Bearer {token}

Response:
{
  "unknown_users": [
    {
      "unknown_id": "UNKNOWN_20250814_a1b2c3d4",
      "display_name": "Unknown Person a1b2c3d4",
      "first_seen": "2025-08-14T12:30:00Z",
      "last_seen": "2025-08-14T16:45:00Z",
      "appearance_count": 5,
      "camera_locations": ["camera_001", "camera_003"],
      "face_image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
      "image_expires": "2025-09-13T12:30:00Z"
    }
  ],
  "total_count": 15
}

# Identify Unknown User
POST /api/v1/unknown-users/{unknown_id}/identify
Content-Type: application/json
Authorization: Bearer {token}

{
  "name": "Jane Smith",
  "type": "visitor",  # visitor, member, staff, delivery
  "apartment": "B-205",
  "notes": "Regular visitor, friend of resident"
}

Response:
{
  "success": true,
  "message": "Unknown user successfully identified as Jane Smith",
  "image_cleanup": "completed"
}

# Convert Unknown to Member
POST /api/v1/unknown-users/{unknown_id}/convert-to-member
Content-Type: application/json
Authorization: Bearer {token}

{
  "name": "Jane Smith",
  "apartment": "B-205",
  "email": "jane.smith@email.com",
  "phone": "+1234567892"
}

Response:
{
  "success": true,
  "new_member_id": "MEM_20250814_002",
  "converted_embeddings": 3,
  "message": "Unknown user converted to member successfully"
}
```


#### 8.2.4 Camera Management

```yaml
# Add New Camera
POST /api/v1/cameras
Content-Type: application/json
Authorization: Bearer {token}

{
  "camera_id": "camera_004",
  "name": "Parking Entrance",
  "location": "Parking Lot Gate",
  "rtsp_url": "rtsp://192.168.1.100:554/stream1",
  "resolution": "1080p",
  "fps": 30,
  "active": true
}

# Get Camera Status
GET /api/v1/cameras/{camera_id}/status
Authorization: Bearer {token}

Response:
{
  "camera_id": "camera_001",
  "status": "online",
  "last_frame": "2025-08-14T16:50:00Z",
  "fps": 28,
  "processing": true,
  "detections_today": 45
}

# Update Camera Configuration
PUT /api/v1/cameras/{camera_id}
Content-Type: application/json
Authorization: Bearer {token}

{
  "name": "Main Gate - Updated",
  "location": "Primary Entrance",
  "recognition_threshold": 0.87,
  "active": true
}
```


#### 8.2.5 Alert Management

```yaml
# Get Active Alerts
GET /api/v1/alerts?status=active&priority=high
Authorization: Bearer {token}

Response:
{
  "alerts": [
    {
      "alert_id": "alert_001",
      "type": "new_unknown",
      "priority": "high",
      "timestamp": "2025-08-14T16:55:00Z",
      "camera_id": "camera_001",
      "unknown_id": "UNKNOWN_20250814_x9y8z7",
      "message": "New unknown person detected at Main Entrance",
      "acknowledged": false
    }
  ]
}

# Acknowledge Alert
POST /api/v1/alerts/{alert_id}/acknowledge
Authorization: Bearer {token}

{
  "acknowledged_by": "admin@company.com",
  "notes": "Visitor identified and logged"
}

# Configure Alert Rules
PUT /api/v1/alerts/rules
Content-Type: application/json
Authorization: Bearer {token}

{
  "new_unknown_immediate": true,
  "recurring_unknown_threshold": 5,
  "unauthorized_access_sms": true,
  "email_notifications": ["admin@company.com", "security@company.com"]
}
```


### 8.3 WebSocket Endpoints

#### 8.3.1 Real-time Recognition Stream

```javascript
// Connect to real-time recognition feed
const ws = new WebSocket('ws://localhost:8000/ws/recognition/live');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Recognition event:', data);
    
    // Handle different event types
    switch(data.type) {
        case 'member_detected':
            updateMemberLog(data);
            break;
        case 'unknown_detected':
            showUnknownAlert(data);
            break;
        case 'camera_offline':
            updateCameraStatus(data);
            break;
    }
};

// Send camera control commands
ws.send(JSON.stringify({
    action: 'start_recognition',
    camera_id: 'camera_001'
}));
```


#### 8.3.2 Admin Dashboard Updates

```javascript
// Real-time dashboard updates
const adminWs = new WebSocket('ws://localhost:8000/ws/admin/dashboard');

adminWs.onmessage = function(event) {
    const update = JSON.parse(event.data);
    
    // Update dashboard metrics
    updateStatistics(update.stats);
    updateAlertCount(update.alerts);
    updateCameraStatus(update.cameras);
};
```


***

## 9. Database Schema

### 9.1 Relational Database Schema (PostgreSQL)

#### 9.1.1 Members Table

```sql
CREATE TABLE members (
    id SERIAL PRIMARY KEY,
    member_id VARCHAR(50) UNIQUE NOT NULL,  -- MEM_20250814_001
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE,
    phone VARCHAR(20),
    apartment VARCHAR(20),
    status VARCHAR(20) DEFAULT 'active',  -- active, inactive, suspended
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP,
    total_appearances INTEGER DEFAULT 0,
    created_by VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_members_member_id ON members(member_id);
CREATE INDEX idx_members_email ON members(email);
CREATE INDEX idx_members_status ON members(status);
CREATE INDEX idx_members_apartment ON members(apartment);
```


#### 9.1.2 Unknown Users Table

```sql
CREATE TABLE unknown_users (
    id SERIAL PRIMARY KEY,
    unknown_id VARCHAR(50) UNIQUE NOT NULL,  -- UNKNOWN_20250814_a1b2c3d4
    display_name VARCHAR(100) NOT NULL,
    actual_name VARCHAR(100),  -- Set by admin after identification
    user_type VARCHAR(20) DEFAULT 'unknown',  -- unknown, visitor, staff
    status VARCHAR(20) DEFAULT 'unidentified',  -- unidentified, pending_review, identified
    first_seen TIMESTAMP NOT NULL,
    last_seen TIMESTAMP NOT NULL,
    appearance_count INTEGER DEFAULT 1,
    camera_locations JSON,  -- Array of camera IDs
    face_image_path VARCHAR(500),  -- Path to encrypted image
    image_stored_until TIMESTAMP,  -- Auto-deletion date
    image_encrypted BOOLEAN DEFAULT TRUE,
    admin_notes TEXT,
    assigned_admin VARCHAR(100),
    identified_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_unknown_users_unknown_id ON unknown_users(unknown_id);
CREATE INDEX idx_unknown_users_status ON unknown_users(status);
CREATE INDEX idx_unknown_users_last_seen ON unknown_users(last_seen);
CREATE INDEX idx_unknown_users_appearance_count ON unknown_users(appearance_count);
```


#### 9.1.3 Appearance Logs Table

```sql
CREATE TABLE appearance_logs (
    id SERIAL PRIMARY KEY,
    log_type VARCHAR(20) NOT NULL,  -- member_entry, unknown_detection
    member_id VARCHAR(50),  -- References members.member_id
    unknown_id VARCHAR(50),  -- References unknown_users.unknown_id
    camera_id VARCHAR(50) NOT NULL,
    location VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    detection_confidence DECIMAL(3,2),
    bbox_coordinates JSON,  -- Face bounding box coordinates
    alert_triggered BOOLEAN DEFAULT FALSE,
    processed BOOLEAN DEFAULT TRUE,
    notes TEXT
);

-- Indexes
CREATE INDEX idx_appearance_logs_member_id ON appearance_logs(member_id);
CREATE INDEX idx_appearance_logs_unknown_id ON appearance_logs(unknown_id);
CREATE INDEX idx_appearance_logs_camera_id ON appearance_logs(camera_id);
CREATE INDEX idx_appearance_logs_timestamp ON appearance_logs(timestamp);
CREATE INDEX idx_appearance_logs_log_type ON appearance_logs(log_type);

-- Foreign key constraints
ALTER TABLE appearance_logs 
    ADD CONSTRAINT fk_appearance_member 
    FOREIGN KEY (member_id) REFERENCES members(member_id);

ALTER TABLE appearance_logs 
    ADD CONSTRAINT fk_appearance_unknown 
    FOREIGN KEY (unknown_id) REFERENCES unknown_users(unknown_id);
```


#### 9.1.4 Cameras Table

```sql
CREATE TABLE cameras (
    id SERIAL PRIMARY KEY,
    camera_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    location VARCHAR(100) NOT NULL,
    rtsp_url VARCHAR(500) NOT NULL,
    resolution VARCHAR(20),  -- 720p, 1080p, 4K
    fps INTEGER DEFAULT 30,
    status VARCHAR(20) DEFAULT 'offline',  -- online, offline, error
    last_heartbeat TIMESTAMP,
    recognition_enabled BOOLEAN DEFAULT TRUE,
    recognition_threshold DECIMAL(3,2) DEFAULT 0.85,
    unknown_threshold DECIMAL(3,2) DEFAULT 0.80,
    processing_config JSON,  -- Camera-specific processing settings
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_cameras_camera_id ON cameras(camera_id);
CREATE INDEX idx_cameras_status ON cameras(status);
CREATE INDEX idx_cameras_location ON cameras(location);
```


#### 9.1.5 Alerts Table

```sql
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(50) UNIQUE NOT NULL,
    alert_type VARCHAR(50) NOT NULL,  -- new_unknown, recurring_unknown, unauthorized_access
    priority VARCHAR(20) DEFAULT 'medium',  -- low, medium, high, critical
    status VARCHAR(20) DEFAULT 'active',  -- active, acknowledged, resolved
    camera_id VARCHAR(50),
    member_id VARCHAR(50),
    unknown_id VARCHAR(50),
    message TEXT NOT NULL,
    alert_data JSON,  -- Additional alert-specific data
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR(100),
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(100),
    notes TEXT
);

-- Indexes
CREATE INDEX idx_alerts_alert_type ON alerts(alert_type);
CREATE INDEX idx_alerts_priority ON alerts(priority);
CREATE INDEX idx_alerts_status ON alerts(status);
CREATE INDEX idx_alerts_triggered_at ON alerts(triggered_at);
CREATE INDEX idx_alerts_camera_id ON alerts(camera_id);
```


#### 9.1.6 System Configuration Table

```sql
CREATE TABLE system_config (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value TEXT,
    config_type VARCHAR(20) DEFAULT 'string',  -- string, integer, boolean, json
    description TEXT,
    category VARCHAR(50),  -- recognition, alerts, security, storage
    is_sensitive BOOLEAN DEFAULT FALSE,  -- For encrypted storage
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default configurations
INSERT INTO system_config (config_key, config_value, config_type, description, category) VALUES
('recognition_threshold', '0.85', 'decimal', 'Minimum similarity for member recognition', 'recognition'),
('unknown_threshold', '0.80', 'decimal', 'Minimum similarity for unknown user matching', 'recognition'),
('unknown_image_retention_days', '30', 'integer', 'Days to retain unknown user images', 'storage'),
('max_embeddings_per_user', '5', 'integer', 'Maximum embeddings stored per user', 'recognition'),
('alert_email_enabled', 'true', 'boolean', 'Enable email alerts', 'alerts'),
('auto_cleanup_enabled', 'true', 'boolean', 'Enable automatic image cleanup', 'storage');
```


### 9.2 Vector Database Schema

#### 9.2.1 Member Embeddings (Pinecone/Qdrant)

```python
# Vector database structure for member embeddings
member_embedding_schema = {
    "id": "MEM_20250814_001_embedding_0",  # Unique vector ID
    "vector": [0.23, -0.87, 0.45, ...],     # 128 or 512 dimensional embedding
    "metadata": {
        "member_id": "MEM_20250814_001",
        "name": "John Doe",
        "user_type": "member",
        "embedding_type": "individual",      # individual, centroid
        "photo_index": 0,
        "registration_date": "2025-08-14T10:30:00Z",
        "apartment": "A-101",
        "is_active": True,
        "quality_score": 0.95
    }
}

# Centroid embedding (average of all individual embeddings)
member_centroid_schema = {
    "id": "MEM_20250814_001_centroid",
    "vector": [0.24, -0.86, 0.44, ...],     # Average of all individual embeddings
    "metadata": {
        "member_id": "MEM_20250814_001",
        "name": "John Doe",
        "user_type": "member",
        "embedding_type": "centroid",
        "photo_count": 3,
        "avg_quality_score": 0.93,
        "registration_date": "2025-08-14T10:30:00Z",
        "apartment": "A-101",
        "is_active": True
    }
}
```


#### 9.2.2 Unknown User Embeddings

```python
# Unknown user embedding structure
unknown_embedding_schema = {
    "id": "UNKNOWN_20250814_a1b2c3d4_embedding_0",
    "vector": [0.31, -0.92, 0.38, ...],
    "metadata": {
        "member_id": "UNKNOWN_20250814_a1b2c3d4",  # Using member_id for consistency
        "name": "Unknown Person a1b2c3d4",
        "actual_name": None,                        # Set after identification
        "user_type": "unknown",
        "status": "unidentified",                   # unidentified, identified
        "embedding_type": "individual",
        "appearance_index": 0,
        "first_seen": "2025-08-14T12:30:00Z",
        "last_seen": "2025-08-14T16:45:00Z",
        "appearance_count": 5,
        "camera_locations": ["camera_001", "camera_003"],
        "requires_admin_review": True,
        "is_active": True
    }
}

# Additional embeddings for recurring unknown users
unknown_additional_schema = {
    "id": "UNKNOWN_20250814_a1b2c3d4_embedding_3",
    "vector": [0.29, -0.94, 0.41, ...],
    "metadata": {
        "member_id": "UNKNOWN_20250814_a1b2c3d4",
        "user_type": "unknown",
        "embedding_type": "additional",
        "appearance_index": 3,
        "timestamp": "2025-08-14T16:45:00Z",
        "camera_id": "camera_003",
        "is_active": True
    }
}
```


#### 9.2.3 Vector Database Queries

```python
# Query examples for different operations

# 1. Find similar faces for recognition
def query_for_recognition(query_embedding, threshold=0.85):
    return vector_db.query(
        vector=query_embedding,
        top_k=5,
        filter={
            "is_active": True,
            "user_type": {"$in": ["member", "unknown"]}
        },
        include_metadata=True
    )

# 2. Search only among members
def query_members_only(query_embedding, threshold=0.85):
    return vector_db.query(
        vector=query_embedding,
        top_k=3,
        filter={
            "user_type": "member",
            "is_active": True
        },
        include_metadata=True
    )

# 3. Search only among unknown users
def query_unknown_users(query_embedding, threshold=0.80):
    return vector_db.query(
        vector=query_embedding,
        top_k=3,
        filter={
            "user_type": "unknown",
            "status": {"$in": ["unidentified", "pending_review"]},
            "is_active": True
        },
        include_metadata=True
    )

# 4. Get all embeddings for specific member
def get_member_embeddings(member_id):
    return vector_db.query(
        vector=[^0] * 128,  # Dummy vector, we only want metadata
        top_k=10,
        filter={
            "member_id": member_id,
            "is_active": True
        },
        include_metadata=True,
        include_values=True
    )
```


### 9.3 Redis Cache Schema

#### 9.3.1 Cache Structure

```python
# Cache keys and structures

# 1. Member cache (frequently accessed members)
redis_keys = {
    # Member basic info cache (TTL: 1 hour)
    "member:{member_id}": {
        "name": "John Doe",
        "apartment": "A-101",
        "status": "active",
        "last_seen": "2025-08-14T15:45:00Z"
    },
    
    # Member embeddings cache (TTL: 4 hours)
    "embeddings:{member_id}": {
        "centroid": [0.24, -0.86, ...],
        "individual_count": 3,
        "quality_score": 0.93
    },
    
    # Camera status cache (TTL: 5 minutes)
    "camera:{camera_id}:status": {
        "status": "online",
        "last_frame": "2025-08-14T16:50:00Z",
        "fps": 28,
        "processing": True
    },
    
    # Recognition statistics cache (TTL: 15 minutes)
    "stats:daily": {
        "date": "2025-08-14",
        "total_detections": 85,
        "known_members": 70,
        "unknown_users": 15,
        "accuracy": 0.96
    },
    
    # Recent unknown users cache (TTL: 30 minutes)
    "unknown:recent": [
        "UNKNOWN_20250814_a1b2c3d4",
        "UNKNOWN_20250814_x9y8z7w6",
        "UNKNOWN_20250814_m5n4p3q2"
    ],
    
    # Active alerts cache (TTL: 10 minutes)
    "alerts:active": [
        {
            "alert_id": "alert_001",
            "type": "new_unknown",
            "priority": "high",
            "timestamp": "2025-08-14T16:55:00Z"
        }
    ]
}

# Session storage
# User sessions (TTL: 24 hours)
"session:{user_id}": {
    "username": "admin@company.com",
    "role": "admin",
    "permissions": ["read", "write", "admin"],
    "login_time": "2025-08-14T09:00:00Z",
    "last_activity": "2025-08-14T16:30:00Z"
}

# Background task status
"task:{task_id}": {
    "status": "processing",
    "progress": 75,
    "started_at": "2025-08-14T16:00:00Z",
    "estimated_completion": "2025-08-14T16:05:00Z"
}
```


***

## 10. Security \& Privacy

### 10.1 Data Protection Strategy

#### 10.1.1 Embedding-First Approach

```python
# Core privacy principle: No biometric image storage for known members
class PrivacyManager:
    def register_member(self, member_photos):
        """
        Privacy-preserving member registration
        1. Process photos temporarily in memory
        2. Extract mathematical embeddings
        3. Store only embeddings in vector database
        4. Immediately delete original photos
        """
        embeddings = []
        
        for photo in member_photos:
            # Process in memory only
            face = self.detect_face(photo)
            embedding = self.extract_embedding(face)
            embeddings.append(embedding)
            
            # Photo is automatically garbage collected
            # No permanent storage of biometric image
        
        return self.store_embeddings_only(embeddings)
    
    def handle_unknown_user(self, face_image):
        """
        Temporary encrypted storage for unknown users
        1. Encrypt face image before storage
        2. Set automatic deletion timer
        3. Store only for admin identification
        4. Delete immediately after identification
        """
        encrypted_image = self.encrypt_image(face_image)
        expiry_date = datetime.now() + timedelta(days=30)
        
        return self.store_temp_encrypted(encrypted_image, expiry_date)
```


#### 10.1.2 Encryption Implementation

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class EncryptionService:
    def __init__(self, password: str):
        self.password = password.encode()
        self.salt = os.urandom(16)
        self.fernet = self._generate_key()
    
    def _generate_key(self):
        """Generate encryption key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password))
        return Fernet(key)
    
    def encrypt_image(self, image_bytes: bytes) -> bytes:
        """Encrypt image bytes"""
        return self.fernet.encrypt(image_bytes)
    
    def decrypt_image(self, encrypted_bytes: bytes) -> bytes:
        """Decrypt image bytes"""
        return self.fernet.decrypt(encrypted_bytes)
    
    def secure_delete(self, file_path: str):
        """Securely overwrite and delete file"""
        if os.path.exists(file_path):
            # Overwrite with random data
            with open(file_path, 'r+b') as file:
                length = file.seek(0, 2)
                file.seek(0)
                file.write(os.urandom(length))
                file.flush()
                os.fsync(file.fileno())
            
            # Delete file
            os.remove(file_path)
```


### 10.2 Access Control \& Authentication

#### 10.2.1 Role-Based Access Control (RBAC)

```python
from enum import Enum
from functools import wraps
from fastapi import HTTPException, Depends

class UserRole(Enum):
    ADMIN = "admin"
    SECURITY = "security"
    OPERATOR = "operator"
    VIEWER = "viewer"

class Permission(Enum):
    READ_MEMBERS = "read_members"
    WRITE_MEMBERS = "write_members"
    DELETE_MEMBERS = "delete_members"
    VIEW_UNKNOWN = "view_unknown"
    IDENTIFY_UNKNOWN = "identify_unknown"
    MANAGE_CAMERAS = "manage_cameras"
    VIEW_SYSTEM_LOGS = "view_system_logs"
    ADMIN_SETTINGS = "admin_settings"

# Role-Permission mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.READ_MEMBERS,
        Permission.WRITE_MEMBERS,
        Permission.DELETE_MEMBERS,
        Permission.VIEW_UNKNOWN,
        Permission.IDENTIFY_UNKNOWN,
        Permission.MANAGE_CAMERAS,
        Permission.VIEW_SYSTEM_LOGS,
        Permission.ADMIN_SETTINGS
    ],
    UserRole.SECURITY: [
        Permission.READ_MEMBERS,
        Permission.WRITE_MEMBERS,
        Permission.VIEW_UNKNOWN,
        Permission.IDENTIFY_UNKNOWN,
        Permission.MANAGE_CAMERAS,
        Permission.VIEW_SYSTEM_LOGS
    ],
    UserRole.OPERATOR: [
        Permission.READ_MEMBERS,
        Permission.VIEW_UNKNOWN,
        Permission.IDENTIFY_UNKNOWN
    ],
    UserRole.VIEWER: [
        Permission.READ_MEMBERS,
        Permission.VIEW_UNKNOWN
    ]
}

def require_permission(permission: Permission):
    """Decorator to enforce permission requirements"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = get_current_user()  # From JWT token
            user_permissions = ROLE_PERMISSIONS.get(current_user.role, [])
            
            if permission not in user_permissions:
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission {permission.value} required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Usage in API endpoints
@app.post("/api/v1/members/register")
@require_permission(Permission.WRITE_MEMBERS)
async def register_member(member_data: MemberCreateSchema):
    # Only users with WRITE_MEMBERS permission can register members
    pass

@app.delete("/api/v1/members/{member_id}")
@require_permission(Permission.DELETE_MEMBERS)
async def delete_member(member_id: str):
    # Only users with DELETE_MEMBERS permission can delete members
    pass
```


#### 10.2.2 JWT Implementation

```python
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

class JWTManager:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expiration_hours = 24
    
    def create_access_token(self, user_data: dict) -> str:
        """Create JWT access token"""
        expiration = datetime.utcnow() + timedelta(hours=self.expiration_hours)
        
        payload = {
            "user_id": user_data["user_id"],
            "username": user_data["username"],
            "role": user_data["role"],
            "permissions": user_data["permissions"],
            "exp": expiration,
            "iat": datetime.utcnow(),
            "iss": "face-recognition-system"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current authenticated user"""
    jwt_manager = JWTManager(os.getenv("SECRET_KEY"))
    payload = jwt_manager.verify_token(credentials.credentials)
    
    # Additional validation
    user = await get_user_by_id(payload["user_id"])
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    
    return user
```


### 10.3 GDPR Compliance

#### 10.3.1 Data Subject Rights Implementation

```python
class GDPRComplianceManager:
    def __init__(self, db_session, vector_db, encryption_service):
        self.db = db_session
        self.vector_db = vector_db
        self.encryption = encryption_service
    
    async def handle_data_subject_request(self, request_type: str, subject_id: str):
        """Handle GDPR data subject rights requests"""
        
        if request_type == "access":
            return await self.provide_data_access(subject_id)
        elif request_type == "portability":
            return await self.export_user_data(subject_id)
        elif request_type == "rectification":
            return await self.allow_data_correction(subject_id)
        elif request_type == "erasure":
            return await self.delete_user_data(subject_id)
        elif request_type == "restriction":
            return await self.restrict_processing(subject_id)
    
    async def provide_data_access(self, member_id: str) -> dict:
        """Right of Access - provide all stored data"""
        member_data = await self.db.query(Member).filter_by(member_id=member_id).first()
        
        if not member_data:
            return {"error": "No data found for this subject"}
        
        # Get embeddings metadata (not actual vectors for privacy)
        embeddings_info = self.vector_db.query(
            vector=[^0] * 128,  # Dummy vector
            filter={"member_id": member_id},
            top_k=100,
            include_metadata=True,
            include_values=False  # Don't return actual embedding values
        )
        
        return {
            "personal_data": {
                "name": member_data.name,
                "email": member_data.email,
                "apartment": member_data.apartment,
                "registration_date": member_data.registration_date,
                "last_seen": member_data.last_seen
            },
            "biometric_data": {
                "embeddings_count": len(embeddings_info["matches"]),
                "embedding_creation_dates": [
                    match["metadata"]["registration_date"] 
                    for match in embeddings_info["matches"]
                ],
                "note": "Biometric templates are mathematical representations, not images"
            },
            "activity_logs": await self.get_member_activity_logs(member_id)
        }
    
    async def delete_user_data(self, member_id: str) -> dict:
        """Right to Erasure - delete all user data"""
        try:
            # 1. Delete from relational database
            member = await self.db.query(Member).filter_by(member_id=member_id).first()
            if member:
                await self.db.delete(member)
            
            # 2. Delete appearance logs
            await self.db.query(AppearanceLog).filter_by(member_id=member_id).delete()
            
            # 3. Delete from vector database
            # Get all vector IDs for this member
            member_vectors = self.vector_db.query(
                vector= * 128,
                filter={"member_id": member_id},
                top_k=100,
                include_metadata=True
            )
            
            vector_ids = [match["id"] for match in member_vectors["matches"]]
            if vector_ids:
                self.vector_db.delete(ids=vector_ids)
            
            # 4. Delete any cached data
            await self.clear_member_cache(member_id)
            
            # 5. Log deletion for audit
            await self.log_data_deletion(member_id, "gdpr_erasure_request")
            
            await self.db.commit()
            
            return {
                "success": True,
                "message": f"All data for member {member_id} has been permanently deleted",
                "deleted_items": {
                    "database_records": 1,
                    "biometric_templates": len(vector_ids),
                    "activity_logs": "all",
                    "cache_entries": "all"
                }
            }
            
        except Exception as e:
            await self.db.rollback()
            return {
                "success": False,
                "error": f"Failed to delete user data: {str(e)}"
            }
    
    async def anonymize_unknown_user(self, unknown_id: str) -> dict:
        """Anonymize unknown user data after identification"""
        try:
            # 1. Delete encrypted face image
            unknown_user = await self.db.query(UnknownUser).filter_by(
                unknown_id=unknown_id
            ).first()
            
            if unknown_user.face_image_path:
                self.encryption.secure_delete(unknown_user.face_image_path)
            
            # 2. Update database record to remove identifiable information
            unknown_user.face_image_path = None
            unknown_user.image_encrypted = False
            unknown_user.admin_notes = "Data anonymized after identification"
            unknown_user.updated_at = datetime.now()
            
            # 3. Keep embeddings for system learning but mark as anonymized
            self.vector_db.update(
                id=f"{unknown_id}_embedding_0",
                set_metadata={
                    "anonymized": True,
                    "anonymization_date": datetime.now().isoformat(),
                    "name": "Anonymized User",
                    "requires_admin_review": False
                }
            )
            
            await self.db.commit()
            
            return {
                "success": True,
                "message": "Unknown user data anonymized successfully"
            }
            
        except Exception as e:
            await self.db.rollback()
            return {
                "success": False,
                "error": f"Failed to anonymize user data: {str(e)}"
            }
```


#### 10.3.2 Consent Management

```python
class ConsentManager:
    def __init__(self, db_session):
        self.db = db_session
    
    async def record_consent(self, member_id: str, consent_data: dict):
        """Record user consent for biometric processing"""
        consent_record = ConsentRecord(
            member_id=member_id,
            consent_type="biometric_processing",
            consent_given=consent_data["consent_given"],
            consent_date=datetime.now(),
            consent_version="v1.0",
            ip_address=consent_data.get("ip_address"),
            user_agent=consent_data.get("user_agent"),
            consent_text=consent_data["consent_text"]
        )
        
        self.db.add(consent_record)
        await self.db.commit()
    
    async def withdraw_consent(self, member_id: str):
        """Handle consent withdrawal"""
        # Mark consent as withdrawn
        consent = await self.db.query(ConsentRecord).filter_by(
            member_id=member_id
        ).first()
        
        if consent:
            consent.consent_withdrawn = True
            consent.withdrawal_date = datetime.now()
        
        # Automatically trigger data deletion
        gdpr_manager = GDPRComplianceManager(self.db, vector_db, encryption_service)
        deletion_result = await gdpr_manager.delete_user_data(member_id)
        
        await self.db.commit()
        
        return {
            "consent_withdrawn": True,
            "data_deletion": deletion_result
        }
```


### 10.4 Audit Logging

#### 10.4.1 Comprehensive Audit Trail

```python
from enum import Enum
import json

class AuditAction(Enum):
    MEMBER_REGISTERED = "member_registered"
    MEMBER_UPDATED = "member_updated"
    MEMBER_DELETED = "member_deleted"
    UNKNOWN_DETECTED = "unknown_detected"
    UNKNOWN_IDENTIFIED = "unknown_identified"
    CAMERA_ACCESSED = "camera_accessed"
    DATA_EXPORTED = "data_exported"
    LOGIN_ATTEMPT = "login_attempt"
    PERMISSION_DENIED = "permission_denied"
    SYSTEM_CONFIG_CHANGED = "system_config_changed"

class AuditLogger:
    def __init__(self, db_session):
        self.db = db_session
    
    async def log_action(
        self,
        action: AuditAction,
        user_id: str = None,
        subject_id: str = None,
        details: dict = None,
        ip_address: str = None,
        user_agent: str = None,
        success: bool = True
    ):
        """Log security and privacy relevant actions"""
        
        audit_log = AuditLog(
            action=action.value,
            user_id=user_id,
            subject_id=subject_id,
            details=json.dumps(details) if details else None,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            timestamp=datetime.now()
        )
        
        self.db.add(audit_log)
        await self.db.commit()
        
        # Also log to external security system if configured
        await self.send_to_external_siem(audit_log)
    
    async def log_biometric_processing(
        self,
        member_id: str,
        operation: str,
        embedding_count: int,
        quality_score: float
    ):
        """Log biometric data processing activities"""
        await self.log_action(
            action=AuditAction.MEMBER_REGISTERED if operation == "register" else AuditAction.MEMBER_UPDATED,
            subject_id=member_id,
            details={
                "operation": operation,
                "embedding_count": embedding_count,
                "quality_score": quality_score,
                "processing_timestamp": datetime.now().isoformat()
            }
        )
    
    async def log_data_access(
        self,
        user_id: str,
        accessed_data_type: str,
        subject_id: str = None,
        query_parameters: dict = None
    ):
        """Log access to personal data"""
        await self.log_action(
            action=AuditAction.DATA_EXPORTED,
            user_id=user_id,
            subject_id=subject_id,
            details={
                "data_type": accessed_data_type,
                "query_parameters": query_parameters,
                "access_timestamp": datetime.now().isoformat()
            }
        )

# Usage in API endpoints
@app.post("/api/v1/members/register")
async def register_member(member_data: MemberCreateSchema, request: Request):
    audit_logger = AuditLogger(db_session)
    
    try:
        result = await member_service.register_member(member_data)
        
        # Log successful registration
        await audit_logger.log_biometric_processing(
            member_id=result["member_id"],
            operation="register",
            embedding_count=result["embeddings_count"],
            quality_score=result["quality_score"]
        )
        
        return result
        
    except Exception as e:
        # Log failed registration attempt
        await audit_logger.log_action(
            action=AuditAction.MEMBER_REGISTERED,
            details={
                "error": str(e),
                "member_data": member_data.dict(exclude={"photos"})  # Don't log photos
            },
            ip_address=request.client.host,
            success=False
        )
        raise
```


***

## 11. Performance Specifications

### 11.1 System Performance Targets

#### 11.1.1 Recognition Performance

```yaml
Recognition Accuracy:
  Known Members: ≥95% accuracy (False Accept Rate <5%)
  Unknown Detection: ≥90% accuracy (False Negative Rate <10%)
  Face Detection: ≥98% detection rate in good lighting

Processing Speed:
  Single Frame Processing: <200ms per frame
  Concurrent Recognition: 10+ simultaneous streams
  Database Query: <50ms average response time
  Vector Similarity Search: <100ms for 100k embeddings

Throughput:
  CCTV Processing: 15-30 FPS per camera
  Maximum Cameras: 20+ concurrent streams
  Embedding Generation: 10+ faces per second
  Alert Processing: <5 second notification delay
```


#### 11.1.2 Hardware Performance Requirements

```yaml
Minimum Configuration:
  CPU Usage: <70% during peak processing
  Memory Usage: <8GB RAM for 5 cameras
  GPU Utilization: <80% with NVIDIA GTX 1660 Ti
  Disk I/O: <100MB/s sustained throughput
  Network: <500Mbps bandwidth per 10 cameras

Recommended Production:
  CPU Usage: <50% during peak processing  
  Memory Usage: <32GB RAM for 20 cameras
  GPU Utilization: <60% with NVIDIA RTX 4080
  Disk I/O: <1GB/s sustained throughput
  Network: <2Gbps bandwidth for full deployment
```


### 11.2 Scalability Architecture

#### 11.2.1 Horizontal Scaling Strategy

```python
# Load balancing for recognition processing
class LoadBalancer:
    def __init__(self):
        self.processing_nodes = []
        self.current_loads = {}
    
    def add_processing_node(self, node_id: str, capacity: int):
        """Add new processing node to cluster"""
        self.processing_nodes.append({
            "node_id": node_id,
            "capacity": capacity,
            "current_load": 0,
            "status": "active"
        })
    
    def assign_camera_stream(self, camera_id: str) -> str:
        """Assign camera to least loaded node"""
        available_nodes = [
            node for node in self.processing_nodes 
            if node["status"] == "active" and node["current_load"] < node["capacity"]
        ]
        
        if not available_nodes:
            raise Exception("No available processing capacity")
        
        # Find node with lowest load percentage
        best_node = min(available_nodes, key=lambda x: x["current_load"] / x["capacity"])
        best_node["current_load"] += 1
        
        return best_node["node_id"]

# Distributed processing with Celery
from celery import Celery

app = Celery('face_recognition_workers')

@app.task
def process_camera_frame_task(camera_id: str, frame_data: bytes):
    """Distributed frame processing task"""
    recognition_engine = RecognitionEngine()
    results = recognition_engine.process_frame(frame_data, camera_id)
    
    # Store results and trigger alerts
    store_recognition_results.delay(results)
    
    return results

@app.task  
def store_recognition_results(results: dict):
    """Store recognition results in database"""
    db_session = get_database_session()
    
    for detection in results["detections"]:
        log_entry = AppearanceLog(
            log_type=detection["type"],
            member_id=detection.get("member_id"),
            unknown_id=detection.get("unknown_id"),
            camera_id=results["camera_id"],
            timestamp=results["timestamp"],
            detection_confidence=detection["confidence"]
        )
        db_session.add(log_entry)
    
    db_session.commit()
```


#### 11.2.2 Database Optimization

```python
# Optimized database queries with proper indexing
class OptimizedMemberRepository:
    def __init__(self, db_session):
        self.db = db_session
    
    async def get_active_members_with_cache(self, limit: int = 1000):
        """Get active members with Redis caching"""
        cache_key = f"active_members:{limit}"
        cached_result = await redis_client.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        
        # Optimized query with proper indexes
        query = """
        SELECT member_id, name, apartment, status 
        FROM members 
        WHERE status = 'active' 
        ORDER BY last_seen DESC 
        LIMIT %s
        """
        
        result = await self.db.execute(query, (limit,))
        members = result.fetchall()
        
        # Cache for 1 hour
        await redis_client.setex(cache_key, 3600, json.dumps(members))
        
        return members
    
    async def get_member_appearance_stats(self, member_id: str, days: int = 30):
        """Optimized appearance statistics query"""
        # Use materialized view for better performance
        query = """
        SELECT 
            DATE(timestamp) as date,
            COUNT(*) as appearances,
            COUNT(DISTINCT camera_id) as unique_cameras
        FROM appearance_logs 
        WHERE member_id = %s 
        AND timestamp >= NOW() - INTERVAL '%s days'
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
        """
        
        result = await self.db.execute(query, (member_id, days))
        return result.fetchall()

# Vector database optimization
class OptimizedVectorOperations:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.embedding_cache = {}
    
    async def batch_similarity_search(self, query_embeddings: List[List[float]]):
        """Batch processing for multiple embeddings"""
        results = []
        
        # Process in batches to optimize memory usage
        batch_size = 32
        for i in range(0, len(query_embeddings), batch_size):
            batch = query_embeddings[i:i + batch_size]
            
            # Parallel processing of batch
            batch_tasks = [
                self.single_similarity_search(embedding) 
                for embedding in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results
    
    async def precompute_member_centroids(self):
        """Precompute and cache member centroids for faster matching"""
        all_members = await self.get_all_active_members()
        
        for member in all_members:
            member_embeddings = await self.get_member_embeddings(member["member_id"])
            
            if len(member_embeddings) > 1:
                # Calculate centroid
                centroid = np.mean(member_embeddings, axis=0)
                
                # Cache centroid
                self.embedding_cache[member["member_id"]] = centroid.tolist()
                
                # Store in vector database
                await self.vector_db.upsert(
                    vectors=[{
                        "id": f"{member['member_id']}_centroid",
                        "values": centroid.tolist(),
                        "metadata": {
                            "member_id": member["member_id"],
                            "embedding_type": "centroid",
                            "computed_date": datetime.now().isoformat()
                        }
                    }]
                )
```

 