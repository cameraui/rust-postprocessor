# @camera.ui/rust-postprocessor

Native Rust detection post-processing for the [camera.ui](https://github.com/cameraui/camera.ui) ecosystem: NMS, IoU, box merging, and a world model per camera that tracks objects, notices when they settle and move again, filters by zone and reports line crossings.

Built with [napi-rs](https://napi.rs). Prebuilt binaries ship for macOS and Windows (x64, arm64), Linux (x64 and arm64 with glibc or musl, riscv64 with glibc) and FreeBSD (x64), so there is no compile step on install.

## Installation

```bash
npm install @camera.ui/rust-postprocessor
```

## Usage

```ts
import { CameraWorld, merge, nms } from '@camera.ui/rust-postprocessor';

// greedy non-maximum suppression per label, boxes normalized to 0..1
const kept = nms(detections, 0.5);

// cluster nearby or overlapping same-label boxes into union boxes
const merged = merge(kept, 0.5, 0.02);

// one world per camera, fed every detector frame in time order
const world = new CameraWorld();
world.setMinConfidence(0.5);
const { tracked, events, crossings } = world.ingest(Date.now(), merged);
```

`events` carry an `eventType` of `objectEntered`, `objectSettled`, `objectWoke`, `objectLost`, `objectRecovered`, `objectDeparted` or `bestShotUpdated`.

A detection cut off by the edge of a crop can be passed with `clipped: true`. The world keeps such a box on a track it already has, but starts no new track on it.

`CameraWorld` also takes crossing lines (`setLines`), detection zones and privacy masks (`setZones`, `filterIndices`), confidence thresholds per label (`setMinConfidences`), a camera's own detector reports (`attest`) and camera moves (`notifyCameraMove`). `nmsIndices` and `boxIou` are exported as well. See `index.d.ts` for the full API.

## Development

```bash
npm install
npm run build        # release build (napi build --platform --release)
npm run build:debug  # debug build
npm test             # cargo test --release
npm run lint         # cargo clippy + eslint
```

---

_Part of camera.ui, the local-first camera and NVR platform._
