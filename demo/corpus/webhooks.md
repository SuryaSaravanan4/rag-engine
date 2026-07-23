# Acme Payments API — Webhooks

Webhooks let Acme notify your server when events happen asynchronously — a
charge succeeding, a dispute opening, a payout landing — instead of you polling
the API.

## Configuring an endpoint

Register an HTTPS URL in the Dashboard under Developers → Webhooks, or via
`POST /v1/webhook_endpoints`. Acme sends a `POST` request with a JSON `Event`
object to that URL whenever a subscribed event occurs.

Every event has a `type` (e.g. `charge.succeeded`, `charge.failed`,
`refund.created`, `dispute.opened`) and a `data.object` containing the affected
resource.

## Verifying signatures

Every webhook request includes an `Acme-Signature` header. It is an HMAC-SHA256
of the raw request body, keyed with your endpoint's **signing secret**
(`whsec_...`, shown once when the endpoint is created).

To verify: recompute the HMAC over the *raw* body (not the parsed JSON — parsing
and re-serializing changes the bytes and breaks the signature) and compare it to
the header using a constant-time comparison. Reject the request if they differ.
This is what stops an attacker from forging events.

## Responding

Return a `2xx` status quickly (within 10 seconds) to acknowledge receipt. Do the
actual work asynchronously — if your handler is slow, Acme's delivery times out.

## Retries and ordering

If your endpoint does not return `2xx`, Acme retries with exponential backoff
for up to **3 days**, then marks the event as failed. Events are **not
guaranteed to arrive in order**, and an event may be delivered **more than
once**, so handlers must be idempotent — key off the event `id`, which is
stable across retries.
