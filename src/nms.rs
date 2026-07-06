//! Greedy per-class NMS. SoA layout lets the inner IoU loop process eight
//! boxes at once via `wide::f32x8`. Output is sorted by confidence descending.

use wide::f32x8;

use crate::types::Detection;

/// Returns indices into the original (pre-sort) input array.
fn nms_core(
  detections: &[Detection],
  iou_threshold: f32,
  max_detections: Option<usize>,
) -> Vec<usize> {
  if detections.is_empty() {
    return Vec::new();
  }

  let n = detections.len();
  let max_det = max_detections.unwrap_or(n);

  // Sort indices, not Detections, so we can map back to original positions.
  let mut order: Vec<usize> = (0..n).collect();

  let prefilter_cap = max_det.saturating_mul(10).max(64);
  if order.len() > prefilter_cap {
    order.select_nth_unstable_by(prefilter_cap, |&a, &b| {
      detections[b]
        .confidence
        .partial_cmp(&detections[a].confidence)
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(prefilter_cap);
  }

  order.sort_unstable_by(|&a, &b| {
    detections[b]
      .confidence
      .partial_cmp(&detections[a].confidence)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  let m = order.len();

  let mut x1 = vec![0.0f32; m];
  let mut y1 = vec![0.0f32; m];
  let mut x2 = vec![0.0f32; m];
  let mut y2 = vec![0.0f32; m];
  let mut areas = vec![0.0f32; m];

  for (i, &orig) in order.iter().enumerate() {
    let det = &detections[orig];
    x1[i] = det.x;
    y1[i] = det.y;
    x2[i] = det.x + det.width;
    y2[i] = det.y + det.height;
    areas[i] = det.width * det.height;
  }

  let mut label_ids = vec![0u32; m];
  let mut label_table: Vec<&str> = Vec::new();
  for (i, &orig) in order.iter().enumerate() {
    let id = match label_table
      .iter()
      .position(|l| *l == detections[orig].label.as_str())
    {
      Some(idx) => idx as u32,
      None => {
        label_table.push(detections[orig].label.as_str());
        (label_table.len() - 1) as u32
      }
    };
    label_ids[i] = id;
  }

  let mut suppressed = vec![false; m];
  let mut keep_sorted: Vec<usize> = Vec::with_capacity(max_det.min(m));

  let iou_v = f32x8::splat(iou_threshold);
  let zero_v = f32x8::splat(0.0);

  for i in 0..m {
    if suppressed[i] {
      continue;
    }
    keep_sorted.push(i);
    if keep_sorted.len() >= max_det {
      break;
    }

    let ax1 = f32x8::splat(x1[i]);
    let ay1 = f32x8::splat(y1[i]);
    let ax2 = f32x8::splat(x2[i]);
    let ay2 = f32x8::splat(y2[i]);
    let aa = f32x8::splat(areas[i]);
    let a_label = label_ids[i];

    let mut j = i + 1;
    while j + 8 <= m {
      let mut chunk_active = false;
      for k in 0..8 {
        if !suppressed[j + k] && label_ids[j + k] == a_label {
          chunk_active = true;
          break;
        }
      }

      if chunk_active {
        let bx1: f32x8 = unsafe { (x1.as_ptr().add(j) as *const f32x8).read_unaligned() };
        let by1: f32x8 = unsafe { (y1.as_ptr().add(j) as *const f32x8).read_unaligned() };
        let bx2: f32x8 = unsafe { (x2.as_ptr().add(j) as *const f32x8).read_unaligned() };
        let by2: f32x8 = unsafe { (y2.as_ptr().add(j) as *const f32x8).read_unaligned() };
        let ba: f32x8 = unsafe { (areas.as_ptr().add(j) as *const f32x8).read_unaligned() };

        let ix1 = ax1.fast_max(bx1);
        let iy1 = ay1.fast_max(by1);
        let ix2 = ax2.fast_min(bx2);
        let iy2 = ay2.fast_min(by2);

        let iw = (ix2 - ix1).fast_max(zero_v);
        let ih = (iy2 - iy1).fast_max(zero_v);
        let inter = iw * ih;
        let union = aa + ba - inter;
        let safe_union = union.fast_max(f32x8::splat(f32::MIN_POSITIVE));
        let iou = inter / safe_union;

        let mask = iou.simd_gt(iou_v);
        let bits = mask.to_bitmask();
        if bits != 0 {
          for k in 0..8 {
            if (bits & (1 << k)) != 0 && !suppressed[j + k] && label_ids[j + k] == a_label {
              suppressed[j + k] = true;
            }
          }
        }
      }
      j += 8;
    }

    while j < m {
      if !suppressed[j] && label_ids[j] == a_label {
        let ix1 = x1[i].max(x1[j]);
        let iy1 = y1[i].max(y1[j]);
        let ix2 = x2[i].min(x2[j]);
        let iy2 = y2[i].min(y2[j]);
        let iw = (ix2 - ix1).max(0.0);
        let ih = (iy2 - iy1).max(0.0);
        let inter = iw * ih;
        let union = areas[i] + areas[j] - inter;
        if union > 0.0 && inter / union > iou_threshold {
          suppressed[j] = true;
        }
      }
      j += 1;
    }
  }

  keep_sorted.iter().map(|&si| order[si]).collect()
}

/// Surviving detections, sorted by confidence descending.
pub fn nms(
  detections: Vec<Detection>,
  iou_threshold: f32,
  max_detections: Option<usize>,
) -> Vec<Detection> {
  let indices = nms_core(&detections, iou_threshold, max_detections);
  indices.into_iter().map(|i| detections[i].clone()).collect()
}

/// Indices of surviving detections, sorted by confidence descending.
pub fn nms_indices(detections: &[Detection], iou_threshold: f32) -> Vec<usize> {
  nms_core(detections, iou_threshold, None)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn det(x: f32, y: f32, w: f32, h: f32, conf: f32, label: &str) -> Detection {
    Detection {
      x,
      y,
      width: w,
      height: h,
      confidence: conf,
      label: label.to_string(),
    }
  }

  #[test]
  fn empty_input() {
    assert!(nms(Vec::new(), 0.5, None).is_empty());
  }

  #[test]
  fn single_box_kept() {
    let input = vec![det(0.1, 0.1, 0.2, 0.2, 0.9, "person")];
    let out = nms(input, 0.5, None);
    assert_eq!(out.len(), 1);
  }

  #[test]
  fn duplicate_boxes_suppressed() {
    let input = vec![
      det(0.1, 0.1, 0.2, 0.2, 0.9, "person"),
      det(0.1, 0.1, 0.2, 0.2, 0.8, "person"),
      det(0.1, 0.1, 0.2, 0.2, 0.7, "person"),
    ];
    let out = nms(input, 0.5, None);
    assert_eq!(out.len(), 1);
    assert!((out[0].confidence - 0.9).abs() < 1e-6);
  }

  #[test]
  fn different_classes_kept() {
    let input = vec![
      det(0.1, 0.1, 0.2, 0.2, 0.9, "person"),
      det(0.1, 0.1, 0.2, 0.2, 0.8, "car"),
    ];
    let out = nms(input, 0.5, None);
    assert_eq!(out.len(), 2);
  }

  #[test]
  fn many_overlapping_simd_path() {
    // 20 near-identical boxes to exercise the f32x8 chunked loop.
    let mut input = Vec::new();
    for i in 0..20 {
      input.push(det(0.1, 0.1, 0.2, 0.2, 0.9 - i as f32 * 0.001, "person"));
    }
    let out = nms(input, 0.5, None);
    assert_eq!(out.len(), 1);
  }

  #[test]
  fn max_detections_cap() {
    let input = vec![
      det(0.0, 0.0, 0.1, 0.1, 0.9, "a"),
      det(0.2, 0.2, 0.1, 0.1, 0.85, "b"),
      det(0.4, 0.4, 0.1, 0.1, 0.8, "c"),
      det(0.6, 0.6, 0.1, 0.1, 0.75, "d"),
    ];
    let out = nms(input, 0.5, Some(2));
    assert_eq!(out.len(), 2);
    assert!((out[0].confidence - 0.9).abs() < 1e-6);
    assert!((out[1].confidence - 0.85).abs() < 1e-6);
  }

  #[test]
  fn output_sorted_by_confidence() {
    let input = vec![
      det(0.0, 0.0, 0.1, 0.1, 0.5, "a"),
      det(0.2, 0.2, 0.1, 0.1, 0.9, "b"),
      det(0.4, 0.4, 0.1, 0.1, 0.7, "c"),
    ];
    let out = nms(input, 0.5, None);
    assert_eq!(out.len(), 3);
    assert!(out[0].confidence > out[1].confidence);
    assert!(out[1].confidence > out[2].confidence);
  }
}
