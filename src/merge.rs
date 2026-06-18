//! Merges clusters of nearby/overlapping same-label detections into union boxes.
//! Unlike NMS (which discards rivals), this collapses split detections of one object.

use std::collections::HashMap;

use crate::types::Detection;

/// Merge nearby/overlapping same-label detections via union-find clustering.
pub fn merge_detections(
  detections: Vec<Detection>,
  iou_threshold: f32,
  close_threshold: f32,
) -> Vec<Detection> {
  if detections.is_empty() {
    return Vec::new();
  }

  // Fast path: skip HashMap grouping when everything shares one label.
  let all_same_label = detections.windows(2).all(|w| w[0].label == w[1].label);

  let mut result: Vec<Detection> = Vec::with_capacity(detections.len());

  if all_same_label {
    let indices: Vec<usize> = (0..detections.len()).collect();
    merge_cluster(
      &detections,
      &indices,
      iou_threshold,
      close_threshold,
      &mut result,
    );
    return result;
  }

  let mut by_label: HashMap<String, Vec<usize>> = HashMap::new();
  for (i, det) in detections.iter().enumerate() {
    by_label.entry(det.label.clone()).or_default().push(i);
  }

  for (_label, indices) in by_label {
    merge_cluster(
      &detections,
      &indices,
      iou_threshold,
      close_threshold,
      &mut result,
    );
  }

  result
}

fn merge_cluster(
  detections: &[Detection],
  indices: &[usize],
  iou_threshold: f32,
  close_threshold: f32,
  out: &mut Vec<Detection>,
) {
  let n = indices.len();

  if n == 1 {
    out.push(detections[indices[0]].clone());
    return;
  }

  // SoA layout [x1, y1, x2, y2, area] per detection for cache-friendly compares.
  let mut boxes = vec![0.0f32; n * 5];
  for (i, &orig_idx) in indices.iter().enumerate() {
    let det = &detections[orig_idx];
    let off = i * 5;
    boxes[off] = det.x;
    boxes[off + 1] = det.y;
    boxes[off + 2] = det.x + det.width;
    boxes[off + 3] = det.y + det.height;
    boxes[off + 4] = det.width * det.height;
  }

  let mut parent: Vec<usize> = (0..n).collect();
  fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    x
  }
  fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    parent[ra] = rb;
  }

  for i in 0..n {
    let off_i = i * 5;
    let ix1 = boxes[off_i];
    let iy1 = boxes[off_i + 1];
    let ix2 = boxes[off_i + 2];
    let iy2 = boxes[off_i + 3];
    let i_area = boxes[off_i + 4];

    for j in (i + 1)..n {
      let off_j = j * 5;

      let close = (ix1 - boxes[off_j]).abs() <= close_threshold
        && (iy1 - boxes[off_j + 1]).abs() <= close_threshold;
      if close {
        union(&mut parent, i, j);
        continue;
      }

      let inter_x1 = ix1.max(boxes[off_j]);
      let inter_y1 = iy1.max(boxes[off_j + 1]);
      let inter_x2 = ix2.min(boxes[off_j + 2]);
      let inter_y2 = iy2.min(boxes[off_j + 3]);
      if inter_x2 <= inter_x1 || inter_y2 <= inter_y1 {
        continue;
      }
      let inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
      let union_area = i_area + boxes[off_j + 4] - inter_area;
      if union_area <= 0.0 {
        continue;
      }
      let iou = inter_area / union_area;
      if iou > iou_threshold {
        union(&mut parent, i, j);
      }
    }
  }

  let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
  for i in 0..n {
    let root = find(&mut parent, i);
    clusters.entry(root).or_default().push(i);
  }

  for (_root, members) in clusters {
    if members.len() == 1 {
      out.push(detections[indices[members[0]]].clone());
      continue;
    }

    let mut min_x = 1.0f32;
    let mut min_y = 1.0f32;
    let mut max_x = 0.0f32;
    let mut max_y = 0.0f32;
    let mut max_conf = 0.0f32;

    for &m in &members {
      let off = m * 5;
      if boxes[off] < min_x {
        min_x = boxes[off];
      }
      if boxes[off + 1] < min_y {
        min_y = boxes[off + 1];
      }
      if boxes[off + 2] > max_x {
        max_x = boxes[off + 2];
      }
      if boxes[off + 3] > max_y {
        max_y = boxes[off + 3];
      }
      let conf = detections[indices[m]].confidence;
      if conf > max_conf {
        max_conf = conf;
      }
    }

    min_x = min_x.max(0.0);
    min_y = min_y.max(0.0);
    max_x = max_x.min(1.0);
    max_y = max_y.min(1.0);

    let w = max_x - min_x;
    let h = max_y - min_y;
    if w <= 0.0 || h <= 0.0 {
      continue;
    }

    let template = &detections[indices[members[0]]];
    out.push(Detection {
      x: min_x,
      y: min_y,
      width: w,
      height: h,
      confidence: max_conf,
      label: template.label.clone(),
    });
  }
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
    assert!(merge_detections(Vec::new(), 0.5, 0.1).is_empty());
  }

  #[test]
  fn single_detection_unchanged() {
    let input = vec![det(0.1, 0.1, 0.2, 0.2, 0.9, "person")];
    let out = merge_detections(input, 0.5, 0.1);
    assert_eq!(out.len(), 1);
    assert!((out[0].x - 0.1).abs() < 1e-6);
    assert!((out[0].confidence - 0.9).abs() < 1e-6);
  }

  #[test]
  fn overlapping_boxes_merged_to_union() {
    let input = vec![
      det(0.1, 0.1, 0.2, 0.2, 0.7, "person"),
      det(0.15, 0.15, 0.2, 0.2, 0.9, "person"),
    ];
    let out = merge_detections(input, 0.01, 0.001);
    assert_eq!(out.len(), 1);
    let m = &out[0];
    assert!((m.x - 0.1).abs() < 1e-6);
    assert!((m.y - 0.1).abs() < 1e-6);
    assert!((m.width - 0.25).abs() < 1e-6);
    assert!((m.height - 0.25).abs() < 1e-6);
    assert!((m.confidence - 0.9).abs() < 1e-6);
  }

  #[test]
  fn close_corners_merged_even_without_iou() {
    let input = vec![
      det(0.10, 0.10, 0.05, 0.05, 0.8, "person"),
      det(0.12, 0.12, 0.05, 0.05, 0.7, "person"),
    ];
    let out = merge_detections(input, 0.5, 0.05);
    assert_eq!(out.len(), 1);
  }

  #[test]
  fn different_labels_not_merged() {
    let input = vec![
      det(0.1, 0.1, 0.2, 0.2, 0.7, "person"),
      det(0.1, 0.1, 0.2, 0.2, 0.9, "car"),
    ];
    let out = merge_detections(input, 0.01, 0.001);
    assert_eq!(out.len(), 2);
  }

  #[test]
  fn distant_boxes_not_merged() {
    let input = vec![
      det(0.1, 0.1, 0.05, 0.05, 0.7, "person"),
      det(0.8, 0.8, 0.05, 0.05, 0.9, "person"),
    ];
    let out = merge_detections(input, 0.5, 0.01);
    assert_eq!(out.len(), 2);
  }

  #[test]
  fn three_box_chain() {
    // a-b and b-c overlap but a-c don't; union-find chains all three.
    let input = vec![
      det(0.10, 0.10, 0.10, 0.10, 0.5, "person"),
      det(0.15, 0.15, 0.10, 0.10, 0.6, "person"),
      det(0.20, 0.20, 0.10, 0.10, 0.7, "person"),
    ];
    let out = merge_detections(input, 0.01, 0.001);
    assert_eq!(out.len(), 1);
    let m = &out[0];
    assert!((m.x - 0.1).abs() < 1e-6);
    assert!((m.width - 0.2).abs() < 1e-6);
  }
}
