# Data Service API Documentation

## Overview

The Data Service provides RESTful APIs for data ingestion, transformation, and retrieval.
Base URL: https://api.dataservice.example.com/v2

## 认证 Authentication

### Token Endpoint

POST /auth/token

Request:
```json
{
  "client_id": "your_client_id",
  "client_secret": "your_client_secret",
  "grant_type": "client_credentials"
}
```

Response:
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIs...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

### Token Refresh

POST /auth/refresh

Headers: Authorization: Bearer {refresh_token}

The refresh token is valid for 30 days. After expiration, re-authenticate with credentials.

## 数据格式 Data Schema

### Input Format

All data submissions must conform to the following JSON schema:

```json
{
  "type": "object",
  "required": ["id", "timestamp", "payload"],
  "properties": {
    "id": {"type": "string", "format": "uuid"},
    "timestamp": {"type": "string", "format": "date-time"},
    "payload": {
      "type": "object",
      "properties": {
        "type": {"type": "string", "enum": ["event", "metric", "log"]},
        "data": {"type": "object"}
      }
    }
  }
}
```

### Batch Upload

POST /data/batch

Maximum batch size: 1000 records per request.
Maximum payload: 5MB per request.

Headers:
- Content-Type: application/json
- Authorization: Bearer {access_token}

## Rate Limiting

- Standard tier: 100 requests/minute
- Premium tier: 1000 requests/minute
- Rate limit headers: X-RateLimit-Remaining, X-RateLimit-Reset

## Error Codes

| Code | Description |
|------|-------------|
| 400  | Invalid request format |
| 401  | Authentication failed |
| 403  | Insufficient permissions |
| 429  | Rate limit exceeded |
| 500  | Internal server error |

## SDK Configuration

### Python SDK

```python
from dataservice import Client

client = Client(
    base_url="https://api.dataservice.example.com/v2",
    client_id="your_id",
    client_secret="your_secret",
    timeout=30,
    retry_count=3,
)
```

### Environment Variables

- DATASERVICE_BASE_URL
- DATASERVICE_CLIENT_ID
- DATASERVICE_CLIENT_SECRET
- DATASERVICE_TIMEOUT (default: 30)
