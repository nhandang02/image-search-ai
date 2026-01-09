# CLIP Service Summary

## Overview
This service converts images and text into 512-dimensional vector embeddings using the OpenAI CLIP model. These embeddings are used for visual search functionality in e-commerce platforms.

## Core Components
- **Framework**: FastAPI (Python)
- **Engine**: PyTorch + Transformers (Hugging Face)
- **Model**: OpenAI CLIP-ViT-B/32
- **Environment**: Dockerized

## Request-Response Flow
1. **Request**: Client sends image (Base64) or text.
2. **Preprocessing**: Image is decoded and resized to 224x224.
3. **Inference**: CLIP model generates a 512-dim embedding.
4. **Normalization**: L2 normalization is applied to the vector.
5. **Response**: 512-dim embedding is returned as a JSON array.

## Performance
- **Text Encoding**: 20-50ms (CPU)
- **Image Encoding**: 50-200ms (CPU)
- **GPU Acceleration**: Approximately 5-10x faster if CUDA is available.

## Integration
The service is designed to work with a NestJS backend and a PostgreSQL database using the `pgvector` extension for similarity search.
