# Build the static app, then run the tiny Node server that serves it and proxies the private
# model Space with HF_TOKEN supplied at runtime:  docker run -e HF_TOKEN=hf_… -p 8080:8080 depthwizard
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npx vite build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production PORT=8080
COPY --from=build /app/dist ./dist
COPY server ./server
EXPOSE 8080
USER node
CMD ["node", "server/serve.mjs"]
