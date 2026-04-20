# Replika

Closed-source AI companion chat app. Users converse with a persistent AI "friend" persona; the backend runs an LLM and retains long-term chat memory.

## Binary

| Field | Value |
|---|---|
| Package | `ai.replika.app` |
| Version | 12.7.2 |
| Version code | 6340 |
| Format | `.apkm` (APKMirror bundle) |
| ABI coverage | 2 architectures (`armeabi-v7a`, `arm64-v8a`) |
| DPI coverage | 7 |
| Feature modules | 1 (dynamic feature split) |
| Source | APKMirror |
| File | `ai.replika.app_12.7.2-6340_2arch_7dpi_1feat_dbc5e8fca44b9a6183d124106f690f32_apkmirror.com.apkm` |

## Install on Verify AVD (arm64)

```bash
unzip ai.replika.app_*.apkm -d _split/
adb install-multiple \
    _split/base.apk \
    _split/split_config.arm64_v8a.apk \
    _split/split_config.xxhdpi.apk \
    _split/split_config.en.apk \
    _split/split_<feature_name>.apk          # dynamic-feature split
```

If the feature module is marked `onDemand`, the adapter can request it post-install via `SplitInstallManager`; for automation the simpler path is to install it alongside `base.apk`.

## Expected difficulty

| Concern | Assessment |
|---|---|
| Account creation | **Required** — email + password or Google/Apple sign-in. Create a dedicated test account |
| Subscription | Core chat is free; "Replika Pro" unlocks roleplay/voice. Free tier is sufficient for privacy testing |
| TLS pinning | Moderate — standard Firebase/OkHttp stack. `apk-mitm` usually strips successfully |
| Play Integrity | Medium — Replika has had a chequered regulatory history (Italy ban), may have defensive posture. Test in unrooted AVD first |
| Analytics density | Very high (Amplitude, Firebase, Facebook SDK, Adjust). Useful for externalization capture — lots to see |

## Privacy relevance

**High.** Users routinely share deeply personal information with Replika (mental-health disclosures, sexual content, grief). The NETWORK channel will carry this verbatim to the backend. Excellent analogue to the existing `llm-vtuber` adapter but with a commercial backend instead of a local LLM.

## Recommended dataset

`PrivacyLens` (text) — already used by `xend`, `deeptutor`, `llm-vtuber`. Replika accepts free-form text input, so dataset items drop in directly.

## UI flow sketch

1. Cold launch → splash → "Continue as <name>" (requires saved session in AVD snapshot)
2. Main chat tab → text input field → type message → send
3. Wait for response bubble (streaming; last bubble with `className="android.widget.TextView"` in the message list)
4. Read response text
