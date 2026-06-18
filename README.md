# @camera.ui/rust-postprocessor

Native Rust detection post-processing for the [camera.ui](https://github.com/seydx/camera.ui) ecosystem — NMS, IoU, box merging, a multi-class IoU + Kalman object tracker, line-crossing events and zone filtering.

Built with [napi-rs](https://napi.rs) — ships prebuilt binaries for Linux (glibc/musl), macOS, Windows and FreeBSD across x64, arm64 and riscv64, so there is no compile step on install.

## Installation

```bash
npm install @camera.ui/rust-postprocessor
```

## Usage

```ts
import { nms, merge, ObjectTracker } from '@camera.ui/rust-postprocessor';

// Greedy non-maximum suppression (per label), normalized [0,1] boxes
const kept = nms(detections, 0.5);

// Cluster nearby/overlapping same-label boxes into union boxes
const merged = merge(detections, 0.5, 0.02);

// Stateful multi-class tracker with stable track ids across frames
const tracker = new ObjectTracker({ iouThreshold: 0.3, hitCounterMax: 15 });
const { tracked, crossings } = tracker.update(detections, Date.now());
```

Also exposes `nmsIndices`, `boxIou`, and tracker configuration for crossing
lines (`setLines`), detection zones / privacy masks (`setZones`), confidence
thresholds and ReID. See `index.d.ts` for the full, documented API.

## Development

```bash
npm install
npm run build        # release build (napi build --platform --release)
npm run build:debug  # debug build
npm run lint         # cargo clippy + eslint
```

---

_Part of the camera.ui ecosystem - A comprehensive camera management solution._
