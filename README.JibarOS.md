# platform_external_llamacpp — JibarOS fork

JibarOS-maintained fork of [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp).

Used by `oird` for every llama-backed OIR capability: `text.complete`, `text.translate`, `text.embed`, `vision.describe` (via libmtmd).

## Why fork

- Reproducible snapshot pins per JibarOS release.
- `Android.bp` wiring that builds `libllama` + `libmtmd` for AOSP `system_ext`.

## Branch model

| Branch | Purpose |
|---|---|
| `main` | Currently tracks `bump-b8720` (latest pinned snapshot). |
| `bump-b8720` | v0.6.x snapshot — upstream tag `b8720` (2026-04-09) + JibarOS `Android.bp`. |
| `bump-<next>` | Future upstream bumps. |

Pinned in [`Jibar-OS/JibarOS/default.xml`](https://github.com/Jibar-OS/JibarOS/blob/main/default.xml) — bump that XML to roll the AOSP tree to a newer snapshot.

## Source layout

- `src/` — vendored upstream llama.cpp source.
- `Android.bp` — JibarOS Android build rules.
- `jni/` — JNI wrappers used by oird.
- `scripts/sync_upstream.sh` — re-pull upstream when bumping.

## See also

- Upstream: [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)
- JibarOS consumer: [`oird`](https://github.com/Jibar-OS/oird)
- Project landing: [`Jibar-OS/JibarOS`](https://github.com/Jibar-OS/JibarOS)
