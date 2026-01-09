# Examples and Integration

## API Usage Examples

### Health Check
```bash
curl http://localhost:8000/health
```

### Encode Text
```bash
curl -X POST http://localhost:8000/encode/text \
  -H "Content-Type: application/json" \
  -d '{"text": "red silk dress"}'
```

### Encode Image (Upload)
```bash
curl -X POST http://localhost:8000/encode/image/upload \
  -F "file=@product.jpg"
```

## NestJS Integration

### Client Implementation
```typescript
async getEmbedding(imageBuffer: Buffer) {
  const base64 = imageBuffer.toString('base64');
  const response = await axios.post('http://clip-service:8000/encode/image', {
    image_base64: base64
  });
  return response.data.embedding;
}
```

### Database Query (PostgreSQL + pgvector)
```sql
SELECT * FROM products 
ORDER BY embedding <-> '[0.1, 0.2, ...]' 
LIMIT 10;
```

## Batch Processing
Batch endpoints should be used for background tasks or bulk updates to reduce round-trip overhead.

```bash
curl -X POST http://localhost:8000/encode/batch-texts \
  -H "Content-Type: application/json" \
  -d '{"texts": ["item 1", "item 2"]}'
```
