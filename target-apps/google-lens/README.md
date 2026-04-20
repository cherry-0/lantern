# Google Lens

Closed-source Google-first-party visual search. Point the camera at an object/text/landmark; Lens returns identification, OCR, translations, shopping results, or reverse-image-search matches.

## Binary

| Field | Value |
|---|---|
| Package | `com.google.ar.lens` |
| Version | 1.18.250731009 |
| Version code | 250731009 |
| Format | `.apk` (single-file — no splits needed) |
| ABI | `arm64-v8a` only |
| DPI | `nodpi` |
| Min API | 23 (Android 6.0) |
| Source | APKMirror |
| File | `com.google.ar.lens_1.18.250731009-250731009_minAPI23(arm64-v8a)(nodpi)_apkmirror.com.apk` |

## Install on Verify AVD (arm64)

Single-file — no extraction needed:

```bash
adb install -r com.google.ar.lens_*.apk
```

## Expected difficulty: **HIGH — may not run at all**

| Concern | Assessment |
|---|---|
| Google account | **Required** — Lens tightly integrates with Google Photos, Search history, and location |
| Play Services dependency | **Hard** — Lens will refuse to launch without a recent Google Play Services; the AVD must use a Play-enabled system image, not plain AOSP |
| TLS pinning | Google-internal — uses proprietary `GmsCore` channels alongside HTTPS. mitmproxy may capture some but not all |
| Play Integrity / hardware attestation | **Very high** — this is Google's own app; it verifies device integrity via `DroidGuard` and may silently disable features or refuse to query on AVDs |
| Camera-only input | Most flows require the live camera. The "gallery" path exists but is sometimes hidden on first-launch tutorials |

## ⚠ Realistic assessment

Historically, Google Lens is one of the hardest first-party Google apps to run under instrumentation. Expect one of:

- App installs but refuses to sign in ("Couldn't sign in — check your connection")
- App signs in but Lens queries silently return empty / "temporary error"
- mitmproxy sees only handshake fragments; actual query payloads travel over binary GmsCore channels we cannot decode

**Recommendation: defer Google Lens to Phase 5 (deferred work).** Attempting it before the stack is mature on simpler targets is likely to burn time without producing a working adapter.

## Privacy relevance (if it can be made to work)

**Extreme.** Images go to Google's Vision API, results tied to the signed-in account, Search and Photos histories are updated. A working adapter would be a premier result. But the cost-to-result ratio strongly favors other targets first.

## Recommended dataset

`HR-VISPR` — already loaded, rich in privacy-tagged scenes/people/objects that Lens would attempt to identify.
