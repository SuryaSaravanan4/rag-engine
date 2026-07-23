# Acme Payments API — Refunds

A **refund** returns money from a settled charge back to the customer's
original payment method.

## Create a refund

`POST /v1/refunds`

| Field | Type | Required | Notes |
|---|---|---|---|
| `charge` | string | yes | The charge to refund (`ch_...`). Must have `status: "succeeded"`. |
| `amount` | integer | no | Amount to refund in cents. Omit to refund the full remaining amount. |
| `reason` | string | no | One of `duplicate`, `fraudulent`, or `requested_by_customer`. |

## Partial and repeated refunds

You can refund part of a charge, and you can issue multiple refunds against the
same charge, as long as the sum of all refunds does not exceed the captured
amount. Attempting to over-refund returns `400` with error code
`charge_already_refunded`.

## Timing

Refunds are submitted to the card network immediately, but the funds take
**5–10 business days** to appear on the customer's statement — this is the
network's timeline, not Acme's, and cannot be accelerated.

## Refunding an uncaptured authorization

You do **not** refund an authorization that was never captured — you *release*
it by calling `POST /v1/charges/:id/release` (or simply letting it expire after
7 days). Refunds only apply to money that actually moved.

## Disputes vs. refunds

A refund is voluntary — you initiate it. A **dispute** (chargeback) is initiated
by the cardholder's bank and is handled through the Disputes API, not Refunds.
Once a charge is disputed you can no longer refund it; the disputed amount is
already held by the network pending resolution.
