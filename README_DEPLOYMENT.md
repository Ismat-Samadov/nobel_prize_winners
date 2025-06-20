# Deployment Guide for Active Learning NER

This guide covers deploying the Active Learning NER application to various platforms, with a focus on Render.

## Architecture

The application consists of two main services:
- **Backend API**: FastAPI application serving the NER models and active learning logic
- **Frontend**: React application providing the user interface

## Render Deployment

### Prerequisites

1. GitHub repository with your code
2. Render account (https://render.com)

### Automatic Deployment

1. **Push code to GitHub** with the provided `render.yaml` configuration

2. **Connect to Render**:
   - Go to Render Dashboard
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect the `render.yaml` file

3. **Services will be created**:
   - `active-learning-ner-api` (Backend API)
   - `active-learning-ner-frontend` (Frontend)

### Manual Deployment

#### Backend API

1. **Create Web Service**:
   - Runtime: Python 3
   - Build Command: `cd api && pip install -r requirements.txt`
   - Start Command: `cd api && uvicorn main:app --host 0.0.0.0 --port $PORT`

2. **Environment Variables**:
   ```
   PYTHONPATH=/opt/render/project/src/api:/opt/render/project/src
   ```

3. **Health Check**:
   - Path: `/health`

#### Frontend

1. **Create Web Service**:
   - Runtime: Node.js
   - Build Command: `cd frontend && npm ci && npm run build`
   - Start Command: `cd frontend && npm install -g serve && serve -s build -l $PORT`

2. **Environment Variables**:
   ```
   REACT_APP_API_URL=https://your-api-service.onrender.com
   ```

## Docker Deployment

### Local Development

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the application
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

### Production Docker

```bash
# Build backend
cd api
docker build -t ner-api .

# Build frontend
cd ../frontend
docker build -t ner-frontend .

# Run services
docker run -p 8000:8000 ner-api
docker run -p 3000:80 ner-frontend
```

## Other Deployment Platforms

### Heroku

1. **Backend (API)**:
   ```bash
   # Create Procfile
   echo "web: uvicorn main:app --host 0.0.0.0 --port \$PORT" > Procfile
   
   # Deploy
   heroku create your-app-api
   git subtree push --prefix=api heroku main
   ```

2. **Frontend**:
   ```bash
   # Create package.json scripts
   "scripts": {
     "build": "react-scripts build",
     "start": "serve -s build -l $PORT"
   }
   
   # Deploy
   heroku create your-app-frontend
   git subtree push --prefix=frontend heroku main
   ```

### AWS (Elastic Beanstalk)

1. **Install EB CLI**:
   ```bash
   pip install awsebcli
   ```

2. **Deploy Backend**:
   ```bash
   cd api
   eb init
   eb create production
   eb deploy
   ```

3. **Deploy Frontend**:
   ```bash
   cd frontend
   eb init
   eb create production-frontend
   eb deploy
   ```

### Google Cloud Platform

1. **Backend (App Engine)**:
   ```yaml
   # app.yaml
   runtime: python39
   entrypoint: uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

2. **Frontend (Firebase Hosting)**:
   ```bash
   npm install -g firebase-tools
   firebase init hosting
   npm run build
   firebase deploy
   ```

## Environment Configuration

### Backend Environment Variables

```bash
# Production
PYTHONPATH=/app/src
LOG_LEVEL=INFO
MODEL_CACHE_SIZE=3
MAX_UPLOAD_SIZE=10MB

# Development
PYTHONPATH=./src
LOG_LEVEL=DEBUG
CORS_ORIGINS=http://localhost:3000
```

### Frontend Environment Variables

```bash
# Production
REACT_APP_API_URL=https://your-api-domain.com
REACT_APP_ENV=production

# Development
REACT_APP_API_URL=http://localhost:8000
REACT_APP_ENV=development
```

## Performance Optimization

### Backend

1. **Model Caching**:
   - Use Redis for model caching in production
   - Implement model warm-up on startup

2. **Database**:
   - Add PostgreSQL for session persistence
   - Implement proper connection pooling

3. **Scaling**:
   - Use Gunicorn with multiple workers
   - Implement async request handling

### Frontend

1. **Build Optimization**:
   - Enable code splitting
   - Optimize bundle size
   - Use CDN for assets

2. **Caching**:
   - Implement service worker
   - Cache API responses
   - Use browser caching

## Security Considerations

1. **API Security**:
   - Implement rate limiting
   - Add CORS configuration
   - Use HTTPS in production
   - Validate all inputs

2. **Authentication** (if needed):
   - JWT tokens
   - OAuth integration
   - Session management

3. **Data Protection**:
   - Encrypt sensitive data
   - Secure file uploads
   - Implement access controls

## Monitoring and Logging

### Application Monitoring

```python
# Add to main.py
import logging
from fastapi import Request
import time

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.4f}s"
    )
    return response
```

### Health Checks

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
```

## Troubleshooting

### Common Issues

1. **CORS Errors**:
   - Check frontend API URL configuration
   - Verify CORS middleware setup

2. **Memory Issues**:
   - Monitor model loading
   - Implement model cleanup
   - Use smaller model variants

3. **Slow Response Times**:
   - Enable caching
   - Optimize model inference
   - Use async processing

### Debug Commands

```bash
# Check service status
curl https://your-api.onrender.com/health

# View logs
heroku logs --tail

# Test local deployment
docker-compose logs api
docker-compose logs frontend
```

## Scaling Strategies

1. **Horizontal Scaling**:
   - Load balancer for API instances
   - Database read replicas
   - CDN for frontend assets

2. **Vertical Scaling**:
   - Increase instance resources
   - Optimize memory usage
   - GPU acceleration for models

3. **Microservices**:
   - Separate model serving
   - Dedicated annotation service
   - Independent scaling

## Cost Optimization

1. **Free Tier Usage**:
   - Render: Free tier for small applications
   - Heroku: Free dyno hours (limited)
   - Vercel: Free for frontend hosting

2. **Resource Management**:
   - Auto-scaling policies
   - Scheduled shutdowns
   - Efficient caching

This deployment guide provides comprehensive instructions for deploying the Active Learning NER application to various platforms with production-ready configurations.