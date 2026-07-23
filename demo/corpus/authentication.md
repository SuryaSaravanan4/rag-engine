# Acme Payments API — Authentication

All requests to the Acme Payments API must be authenticated with an API key.

## API keys

Acme issues two kinds of keys:

- **Secret keys** (`sk_live_...`, `sk_test_...`) — used server-side. Never
  expose a secret key in a browser, mobile app, or public repository.
- **Publishable keys** (`pk_live_...`, `pk_test_...`) — safe to embed in
  client-side code. They can only create tokens, not move money.

Test-mode keys (`sk_test_`, `pk_test_`) operate against the sandbox and never
touch real funds. Live-mode keys (`sk_live_`, `pk_live_`) move real money.

## Authenticating a request

Pass the secret key as a Bearer token in the `Authorization` header:

```
Authorization: Bearer sk_live_51H8xY...
```

Requests without a valid key return `401 Unauthorized`. Requests with a
well-formed but revoked key return `401` with error code `key_revoked`.

## Rotating keys

Keys are rotated from the Dashboard under Developers → API keys. Rotation
issues a new secret and keeps the old one valid for a 24-hour grace period so
in-flight deployments do not break. After 24 hours the old key returns `401`.

## Rate limits

The API allows **100 requests per second** per account in live mode and **25
requests per second** in test mode. Exceeding the limit returns
`429 Too Many Requests` with a `Retry-After` header (in seconds). Clients
should retry with exponential backoff and honor `Retry-After`.
