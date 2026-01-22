# Frontend Docker Setup

This guide explains how to build and run the frontend using Docker.

## Quick Start

### Build and Run

```bash
cd frontend
docker build -t semi-supervised-labeling-frontend .
docker run -d -p 3000:80 --name labeling-frontend semi-supervised-labeling-frontend
```

The frontend will be available at `http://localhost:3000`

### Using Docker Compose (Recommended)

If you have a `docker-compose.yml` at the root level that includes both backend and frontend:

```yaml
version: '3.8'

services:
  backend:
    # ... backend config

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - backend
    environment:
      - VITE_API_URL=http://backend:8000/api/v1
```

Then run:
```bash
docker-compose up --build
```

## Multi-Stage Build

The Dockerfile uses a multi-stage build:

1. **Builder stage**: Uses Node.js to build the React application
2. **Production stage**: Uses nginx to serve the static files

This results in a smaller final image (~25MB vs ~500MB with Node.js).

## Environment Variables

The frontend can be configured using environment variables:

- `VITE_API_URL` - Backend API URL (default: `http://localhost:8000/api/v1`)

Note: Environment variables must be prefixed with `VITE_` to be accessible in the frontend code during build time.

## Nginx Configuration

The included `nginx.conf` provides:

- SPA routing support (all routes serve `index.html`)
- API proxy configuration (routes `/api` to backend)
- Gzip compression
- Static asset caching
- Security headers

## Troubleshooting

### Frontend can't connect to backend

If running in Docker, make sure:
1. Both containers are on the same network
2. The `VITE_API_URL` points to the backend service name (e.g., `http://backend:8000/api/v1`)
3. The nginx proxy configuration is correct

### Build fails

Make sure:
- Node.js version matches (check `package.json` engines if specified)
- All dependencies are listed in `package.json`
- Build cache is cleared: `docker build --no-cache -t ...`

### Port conflicts

Change the port mapping:
```bash
docker run -d -p 8080:80 --name labeling-frontend semi-supervised-labeling-frontend
```

## Production Deployment

For production:

1. **Use specific tags** instead of `latest`
2. **Set proper environment variables**
3. **Configure HTTPS** with SSL certificates
4. **Use reverse proxy** (nginx, Traefik) for SSL termination
5. **Set up monitoring** and health checks

Example production docker-compose:

```yaml
services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./ssl:/etc/nginx/ssl:ro
    environment:
      - VITE_API_URL=https://api.yourdomain.com/api/v1
```
