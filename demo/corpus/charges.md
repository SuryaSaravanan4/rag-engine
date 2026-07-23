# Acme Payments API — Charges

A **charge** represents a single attempt to move money from a customer's
payment method to your account.

## Create a charge

`POST /v1/charges`

| Field | Type | Required | Notes |
|---|---|---|---|
| `amount` | integer | yes | Amount in the currency's smallest unit (cents for USD). Minimum 50 (= $0.50). |
| `currency` | string | yes | Three-letter ISO code, lowercase (`usd`, `eur`, `gbp`). |
| `source` | string | yes | A token (`tok_...`) or saved payment method (`pm_...`). |
| `description` | string | no | Arbitrary text shown on the customer's statement and in the Dashboard. |
| `capture` | boolean | no | Defaults to `true`. If `false`, the charge is only authorized (see below). |

The response includes an `id` (`ch_...`), a `status`, and the `amount` echoed
back. A successful charge has `status: "succeeded"`.

## Authorize now, capture later

Set `capture: false` to place a hold on the funds without moving them. The
charge returns `status: "requires_capture"`. Call `POST /v1/charges/:id/capture`
within **7 days** to settle it; uncaptured authorizations are automatically
released after 7 days and the hold disappears.

You may capture *less* than the authorized amount (for example, if part of an
order is out of stock). You cannot capture *more*.

## Idempotency

Because a network retry could otherwise double-charge a customer, send an
`Idempotency-Key` header (any unique string, e.g. a UUID) on charge creation.
Acme stores the result of the first request for **24 hours** and returns that
same result for any retry with the same key, so retries are safe.

## Failure

A declined card returns `402 Payment Required` with a `decline_code`
(`insufficient_funds`, `card_declined`, `expired_card`, ...). The charge object
still exists with `status: "failed"` so you can inspect why.
