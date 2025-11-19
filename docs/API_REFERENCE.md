## Gemäldeagent API Reference

Complete API documentation for the Gemäldeagent REST API.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. Authentication can be added in production using JWT tokens.

---

## Endpoints

### Health & Info

#### `GET /`

Root endpoint with API information.

**Response:**
```json
{
  "name": "Gemäldeagent API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "endpoints": {...}
}
```

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

---

### Artwork Analysis

#### `POST /api/analyze-image`

Analyze an artwork image and return complete analysis.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| image | File | Yes | Image file (JPEG, PNG, WebP) |
| notes | String | No | Optional notes about the artwork |

**Response:**
```json
{
  "success": true,
  "result": {
    "image_path": "uploaded_image",
    "style_estimation": {
      "epoch": "Impressionism",
      "style": "Landscape Impressionism",
      "confidence": 0.92
    },
    "artist_candidates": [
      {
        "name": "Claude Monet",
        "similarity": 0.83,
        "epoch": "Impressionism",
        "style": "Impressionism"
      }
    ],
    "authenticity_score": 78,
    "condition": {
      "craquele": true,
      "yellowing": false,
      "stains": true,
      "cracks": ["lower_right"],
      "damage_score": 0.22,
      "notes": "Some visible surface cracks and stains in lower right quadrant."
    },
    "valuation": {
      "estimated_value": 45000,
      "min": 35000,
      "max": 70000,
      "confidence": "medium",
      "rationale": "Based on 12 similar sales...",
      "comparable_sales": 12
    },
    "provenance": {
      "reverse_image_hits": 0,
      "auction_history": [],
      "notes": "No matches in local provenance dataset.",
      "sources": []
    },
    "metadata": {
      "source": "user_upload",
      "created_at": "2024-01-15T10:30:00Z",
      "notes": "User suspected: Monet",
      "processing_time_seconds": 2.34
    }
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Error message"
}
```

#### `POST /api/analyze-and-report`

Analyze an artwork and generate a downloadable PDF/DOCX report.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| image | File | Yes | Image file |
| notes | String | No | Optional notes |
| format | String | No | Report format: "pdf" (default) or "docx" |

**Response:**
```json
{
  "success": true,
  "analysis": {...},
  "report_path": "/path/to/report.pdf",
  "download_url": "/api/reports/download?path=/path/to/report.pdf"
}
```

#### `GET /api/reports/download`

Download a generated report file.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| path | String | Yes | Report file path from analyze-and-report response |

**Response:**
- Content-Type: `application/pdf` or `application/vnd.openxmlformats-officedocument.wordprocessingml.document`
- File download

---

### Market Monitoring

#### `GET /api/market/items`

Query market items from monitoring database.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| source | String | No | Filter by source (ebay, willhaben, etc.) |
| is_artwork | Boolean | No | Filter by artwork classification |
| min_price | Float | No | Minimum price filter |
| max_price | Float | No | Maximum price filter |
| limit | Integer | No | Maximum results (default: 100, max: 1000) |
| offset | Integer | No | Results offset (default: 0) |

**Response:**
```json
{
  "items": [
    {
      "id": 1,
      "source": "mock_marketplace",
      "url": "https://example.com/item/12345",
      "title": "Antique Oil Painting - Landscape",
      "description": "Beautiful old oil painting...",
      "price": 450,
      "currency": "EUR",
      "seller": "antique_collector_42",
      "location": "Vienna, Austria",
      "timestamp_found": "2024-01-10T12:00:00Z",
      "image_hash": "abc123...",
      "analysis_json": {...},
      "is_artwork": true
    }
  ],
  "total": 150,
  "limit": 100,
  "offset": 0
}
```

#### `POST /api/monitoring/run-scan`

Manually trigger a market monitoring scan.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| sources | Array[String] | No | Specific sources to scan |
| max_items | Integer | No | Max items to scan per source |

**Response:**
```json
{
  "success": true,
  "items_scanned": 50,
  "artworks_found": 12,
  "alerts_created": 3
}
```

#### `GET /api/monitoring/alerts`

Get alerts from market monitoring.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| notified | Boolean | No | Filter by notification status |
| limit | Integer | No | Maximum results (default: 100) |

**Response:**
```json
{
  "success": true,
  "alerts": [
    {
      "id": 1,
      "item_id": 42,
      "alert_level": "critical",
      "reason": "Potential bargain: Listed at €450, estimated value €4,500",
      "created_at": "2024-01-15T14:30:00Z",
      "notified": false,
      "title": "Antique Oil Painting",
      "url": "https://...",
      "price": 450
    }
  ],
  "count": 5
}
```

---

### Statistics

#### `GET /api/stats/overview`

Get overview statistics for the system.

**Response:**
```json
{
  "total_items": 150,
  "total_artworks": 45,
  "total_alerts": 12,
  "unnotified_alerts": 5
}
```

---

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error responses include a descriptive message:

```json
{
  "detail": "Error description"
}
```

---

## Rate Limiting

Currently, no rate limiting is enforced. In production, implement rate limiting based on IP address or API key.

---

## CORS

CORS is configured to allow requests from:
- `http://localhost:3000`
- `http://localhost:8000`

Configure additional origins in `.env`:

```env
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

---

## Interactive Documentation

Visit `/docs` for interactive Swagger UI documentation.

Visit `/redoc` for ReDoc documentation.
