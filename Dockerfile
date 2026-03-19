# 1. 빌드 단계 (Node.js 환경에서 앱을 최적화하여 추출)
FROM node:20-alpine as build-stage
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# 2. 실행 단계 (가볍고 빠른 웹 서버인 Nginx 사용)
FROM nginx:stable-alpine as production-stage
# 위 빌드 단계에서 생성된 dist 폴더(결과물)만 가져옴
COPY --from=build-stage /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]