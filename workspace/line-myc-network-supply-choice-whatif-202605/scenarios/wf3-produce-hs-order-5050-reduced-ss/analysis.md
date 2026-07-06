# WF3 Analysis

## Scenario

- Scenario: `wf3-produce-hs-order-5050-reduced-ss`
- Run ID: `db_wf3-produce-hs-order-5050-reduced-ss_20260511_180429`
- Intent: high-side production with reduced DTC safety stock and 50/50 ordering

## Facts

- CFR: `95.22%`
- Cut rate: `4.78%`
- Average inventory: `160,590.21`
- Ending inventory: `146,574`
- Peak inventory: `172,579`
- Produced quantity: `46,134`
- Changeovers: `13`
- Changeover time: `21.42`
- Order quantity: `58,616`
- Shipment quantity: `55,812`
- Deployed transfer quantity: `64,154`
- Delivered transfer quantity: `61,679`

Top DC shortfalls:

- `C937`: shortfall `509`, CFR `76.13%`
- `A673`: shortfall `457`, CFR `91.03%`
- `D594`: shortfall `338`, CFR `94.19%`

Shortfall cause mix:

- `4_pdt_gr_mismatch`: `50` SKU, `2,244` shortfall qty
- `residual_other`: `12` SKU, `560` shortfall qty

WF1 baseline for causal comparison:

- `wf1` order quantity: `58,480`
- `wf1` shipment quantity: `56,684`
- `wf1` produced quantity: `68,958`
- `wf1` deployed transfer quantity: `103,019`
- `wf1` delivered transfer quantity: `102,157`
- `wf1` CFR: `96.93%`
- `wf1` cut rate: `3.07%`

## Interpretation

`wf3` service is worse not because its demand base is materially smaller
or larger than `wf1`, but because its replenishment flow is materially
weaker. Relative to `wf1`, order qty is essentially flat, increasing by
only `136` units. But shipment qty drops by `872`, cut qty rises by
`1,008`, produced qty drops by `22,824`, deployed transfer qty drops by
`38,865`, and delivered transfer qty drops by `40,478`.

The classifier points to `pdt/gr mismatch` as the dominant shortfall
mode, not missing safety stock rows and not a large block of deployed but
never shipped items. That means the reduced safety stock is acting more
as a buffer removal than as a direct config failure. In other words, the
network already has replenishment timing friction, and `wf3` leaves less
inventory cushion to absorb it.

This explains the apparent contradiction behind the question “produce to
high-side, order less, why is service worst?” Service is a ratio between
shipment and realized order fulfillment, not a reward for lower nominal
volume. In `wf3`, demand is not meaningfully lower than `wf1`, while the
supply response is much smaller. High-side production intent did not
convert into high realized output once the lower safety-stock signal
reduced downstream pull and inter-location deployment.

## Recommendation

Do not treat `wf3` as the leading compromise option. It creates a weaker
service outcome than `wf2` without delivering a uniquely compelling
inventory advantage. It is more useful as evidence that reducing safety
stock while keeping high-side production is an unfavorable combination in
this network. If the business still wants a reduced-safety-stock path,
the current evidence suggests testing it with the `wf4` policy shape or
first fixing the replenishment timing issue that is surfacing as
`pdt/gr mismatch`.

## PDT/GR Mismatch Deep Dive

The highest-importance `wf3` PDT/GR mismatch DCs are `C937`, `A673`, and
`D594`. All of the material rows below still had positive configured
safety stock, so the failure pattern is not “no buffer configured.” The
issue is that actual replenishment timing ran later than configured lead
time assumptions, and the reduced safety stock left less room to absorb
that delay.

### C937

Receiving lane pattern:

- All identified lead-time-mismatch materials are supplied from `C816`
- Configured total lead time is consistently `8.0` days
- Actual weighted lead time ranges from `8.99` to `15.64` days
- Lead-time gap ranges from `0.99` to `7.64` days

Material details:

| Material | Order Qty | Shipment Qty | Shortfall Qty | Safety Stock | Deploy Qty | Ship Qty | Actual LT | Config LT | LT Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `80893247` | 599 | 399 | 200 | 96.5 | 315 | 249 | 13.3293 | 8.0 | 5.3293 |
| `80775500` | 550 | 382 | 168 | 83.0 | 133 | 133 | 8.9925 | 8.0 | 0.9925 |
| `80825738` | 327 | 255 | 72 | 45.0 | 111 | 111 | 12.9910 | 8.0 | 4.9910 |
| `80793241` | 141 | 119 | 22 | 32.0 | 69 | 69 | 12.2609 | 8.0 | 4.2609 |
| `80793243` | 56 | 38 | 18 | 11.0 | 28 | 28 | 15.6429 | 8.0 | 7.6429 |
| `80893246` | 87 | 69 | 18 | 11.5 | 39 | 39 | 9.7949 | 8.0 | 1.7949 |
| `80843500` | 63 | 53 | 10 | 4.0 | 28 | 28 | 14.5357 | 8.0 | 6.5357 |
| `80775504` | 20 | 19 | 1 | 5.0 | 12 | 12 | 13.1667 | 8.0 | 5.1667 |

Interpretation:

`C937` is the clearest severe lane-timing problem in `wf3`. The biggest
shortfall materials all point to the same lane, `C816 -> C937`, and the
actual total lead time is repeatedly far above the configured `8` days.
This is why `C937` remains a weak point across all scenarios and becomes
especially fragile once DTC safety stock is reduced.

### A673

Receiving lane pattern:

- All identified lead-time-mismatch materials are supplied from `C810`
- Configured total lead time is `8.0` days
- Actual weighted lead time ranges from `9.26` to `9.49` days
- Lead-time gap ranges from `1.26` to `1.49` days

Material details:

| Material | Order Qty | Shipment Qty | Shortfall Qty | Safety Stock | Deploy Qty | Ship Qty | Actual LT | Config LT | LT Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `80775500` | 1315 | 1006 | 309 | 130.0 | 534 | 460 | 9.3913 | 8.0 | 1.3913 |
| `80825738` | 457 | 360 | 97 | 58.5 | 217 | 213 | 9.4930 | 8.0 | 1.4930 |
| `80857020` | 320 | 296 | 24 | 100.0 | 216 | 176 | 9.2557 | 8.0 | 1.2557 |

Interpretation:

`A673` is a narrower but still meaningful mismatch pattern. The gap is
much smaller than `C937`, but it is systematic: the same `C810 -> A673`
lane appears across all impacted materials. Here the service damage is
being created by a consistent roughly `1.3` to `1.5` day delay on a lane
that carries relatively large-volume SKUs.

### D594

Receiving lane pattern:

- All identified lead-time-mismatch materials are supplied from `0386`
- Configured total lead time is `6.0` days
- Actual weighted lead time ranges from `6.73` to `6.83` days
- Lead-time gap ranges from `0.73` to `0.83` days

Material details:

| Material | Order Qty | Shipment Qty | Shortfall Qty | Safety Stock | Deploy Qty | Ship Qty | Actual LT | Config LT | LT Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `80861856` | 430 | 310 | 120 | 117.5 | 372 | 372 | 6.7258 | 6.0 | 0.7258 |
| `80845346` | 605 | 578 | 27 | 519.5 | 1016 | 1016 | 6.8297 | 6.0 | 0.8297 |

Interpretation:

`D594` is the mildest of the three PDT/GR mismatch clusters. The timing
gap exists, but it is under one day and the shortfall is concentrated in
two materials. This looks more like a secondary timing issue than the
primary driver of `wf3` service underperformance.