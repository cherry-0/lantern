# Expensify

Closed-source business expense management SaaS. Users photograph receipts; the app OCRs merchant, date, amount, and category — the same data shape extracted by `budget-lens` / SROIE.

## Binary

| Field | Value |
|---|---|
| Package | `org.me.mobiexpensifyg` |
| Version | 9.3.58-9 |
| Version code | 509035809 |
| Format | `.apkm` (APKMirror bundle) |
| ABI coverage | 4 architectures |
| DPI coverage | 7 |
| Source | APKMirror |
| File | `org.me.mobiexpensifyg_9.3.58-9-509035809_4arch_7dpi_9773f02585b3e805bfe6177715f2b69c_apkmirror.com.apkm` |

## Install on Verify AVD (arm64)

```bash
unzip org.me.mobiexpensifyg_*.apkm -d _split/
adb install-multiple \
    _split/base.apk \
    _split/split_config.arm64_v8a.apk \
    _split/split_config.xxhdpi.apk \
    _split/split_config.en.apk
```

## Expected difficulty

| Concern | Assessment |
|---|---|
| Account | **Required** — email-based signup. Personal (non-enterprise) accounts work and include the SmartScan receipt OCR feature |
| Subscription | Free tier supports SmartScan up to 25 receipts/month — sufficient for a Verify dataset pass |
| TLS pinning | High — React Native app, uses `rn-fetch-blob` + custom interceptors. Expect `apk-mitm` to need a second pass, possibly Frida |
| Play Integrity | Medium-high — financial app; may refuse rooted environments |
| Account verification | Email magic-link login on first run — scriptable only if the AVD can access the test email inbox (or use SMS-less email provider) |

## Privacy relevance

**Very high.** Receipts contain merchant name, address, items purchased, card last-4 digits, loyalty numbers, and timestamps. The app uploads the raw receipt image to Expensify's OCR backend — whatever attributes the image holds are exfiltrated in full. Direct analogue to `budget-lens` but for a commercial pipeline.

## Recommended dataset

`SROIE2019` — already loaded for `budget-lens`. Receipt images drop into the app's "upload from gallery" flow.

## UI flow sketch

1. Launch → login screen (bypass via snapshot with saved session)
2. Bottom tab "Expenses" → `+` FAB → "Scan receipt"
3. Choose "From library" (not camera) → system file picker → select `/sdcard/Download/receipt.jpg`
4. Wait for "SmartScan in progress" to resolve → read merchant / amount / date fields from the edit view
