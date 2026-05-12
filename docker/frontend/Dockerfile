# ================================================================
# Frontend Dockerfile - React TypeScript Dashboard
# Multi-stage: node builder → nginx runtime
# ================================================================
FROM node:20-alpine AS builder

WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci --frozen-lockfile

COPY frontend/ .
RUN npm run build

# ── Runtime: nginx ───────────────────────────────────────────
FROM nginx:1.25-alpine AS runtime

COPY --from=builder /app/dist /usr/share/nginx/html
COPY docker/nginx/nginx.conf /etc/nginx/nginx.conf
COPY docker/nginx/default.conf /etc/nginx/conf.d/default.conf

HEALTHCHECK --interval=30s --timeout=5s \
    CMD wget -q --spider http://localhost:3000/health || exit 1

EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]
