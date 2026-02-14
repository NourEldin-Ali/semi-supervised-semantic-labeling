# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Build-time API base URL for Vite (default: same-origin proxy)
ARG VITE_API_URL=/api/v1
ENV VITE_API_URL=$VITE_API_URL

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration templates (not auto-loaded by nginx)
COPY nginx.conf /etc/nginx/conf.d/https.conf.template
COPY nginx.http.conf /etc/nginx/conf.d/http.conf.template

# Entrypoint to choose HTTP/HTTPS based on cert availability
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Expose ports 80/443
EXPOSE 80 443

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/healthz || exit 1

# Start nginx with cert-aware config selection
CMD ["/docker-entrypoint.sh"]
