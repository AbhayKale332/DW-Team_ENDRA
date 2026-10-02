# Build the static app, then run the tiny Node server that serves it and proxies the private
# model Space with HF_TOKEN supplied at runtime:  docker run -e HF_TOKEN=hf_… -p 8080:8080 depthwizard
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Samples too big for GitHub live on the HF dataset; fetched before the build so samples/index.json lists them.
# ADD writes remote files as 0600 root, unreadable by the node user that serves them, hence the chmod.
ARG SAMPLES_URL=https://huggingface.co/datasets/Abhay-Kale/DW_Samples/resolve/main
ADD ["${SAMPLES_URL}/Cartosat%203%20Sample/Cartosat%203%20Sample.dwproj", "public/samples/Cartosat 3 Sample/Cartosat 3 Sample.dwproj"]
ADD ["${SAMPLES_URL}/Cartosat3-PAN-Sample%28B%26W%29/Cartosat3-PAN-Sample%28B%26W%29.dwproj", "public/samples/Cartosat3-PAN-Sample(B&W)/Cartosat3-PAN-Sample(B&W).dwproj"]
ADD ["${SAMPLES_URL}/Mountainus_Region_C2E/Mountainus_Region_C2E.dwproj", "public/samples/Mountainus_Region_C2E/Mountainus_Region_C2E.dwproj"]
RUN chmod -R a+rX public/samples
RUN npm run build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production PORT=8080
COPY --from=build /app/dist ./dist
COPY server ./server
EXPOSE 8080
USER node
CMD ["node", "server/serve.mjs"]
