use std::collections::HashMap;

use crate::types::Detection;

const FRAGMENT_SHARE: f32 = 0.9;
const FRAGMENT_EDGE: f32 = 0.05;

#[cfg(test)]
pub fn merge_detections(
  detections: Vec<Detection>,
  iou_threshold: f32,
  close_threshold: f32,
) -> Vec<Detection> {
  merge_detections_contained(detections, iou_threshold, close_threshold, &[], 0.0)
}

pub fn merge_detections_contained(
  detections: Vec<Detection>,
  iou_threshold: f32,
  close_threshold: f32,
  contained: &[String],
  min_share: f32,
) -> Vec<Detection> {
  if detections.is_empty() {
    return Vec::new();
  }

  let mut by_label: HashMap<String, Vec<usize>> = HashMap::new();
  for (i, det) in detections.iter().enumerate() {
    by_label.entry(det.label.clone()).or_default().push(i);
  }

  let mut result: Vec<Detection> = Vec::with_capacity(detections.len());
  for (label, indices) in by_label {
    let share = if contained.contains(&label) {
      min_share
    } else {
      0.0
    };
    merge_cluster(
      &detections,
      &indices,
      iou_threshold,
      close_threshold,
      share,
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
  min_share: f32,
  out: &mut Vec<Detection>,
) {
  let n = indices.len();

  if n == 1 {
    out.push(detections[indices[0]].clone());
    return;
  }

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

      let inter_x1 = ix1.max(boxes[off_j]);
      let inter_y1 = iy1.max(boxes[off_j + 1]);
      let inter_x2 = ix2.min(boxes[off_j + 2]);
      let inter_y2 = iy2.min(boxes[off_j + 3]);
      let inter_area = if inter_x2 > inter_x1 && inter_y2 > inter_y1 {
        (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
      } else {
        0.0
      };
      let smaller = i_area.min(boxes[off_j + 4]);
      let share = if smaller > 0.0 {
        inter_area / smaller
      } else {
        0.0
      };
      let contained = min_share <= 0.0 || share >= min_share;

      let close = (ix1 - boxes[off_j]).abs() <= close_threshold
        && (iy1 - boxes[off_j + 1]).abs() <= close_threshold;
      if close {
        if contained {
          union(&mut parent, i, j);
        }
        continue;
      }

      if inter_area <= 0.0 {
        continue;
      }
      if min_share > 0.0 && share >= FRAGMENT_SHARE {
        let (small, big) = if i_area <= boxes[off_j + 4] {
          (off_i, off_j)
        } else {
          (off_j, off_i)
        };
        let tolerance = (boxes[big + 3] - boxes[big + 1]) * FRAGMENT_EDGE;
        if (boxes[small + 1] - boxes[big + 1]).abs() <= tolerance
          || (boxes[small + 3] - boxes[big + 3]).abs() <= tolerance
        {
          union(&mut parent, i, j);
          continue;
        }
      }
      let union_area = i_area + boxes[off_j + 4] - inter_area;
      if union_area <= 0.0 {
        continue;
      }
      if inter_area / union_area > iou_threshold && contained {
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
  #[test]
  fn contained_label_keeps_two_people_apart() {
    // walkout: one lying, one sitting, iou above 0.3 but only half inside
    let input = vec![
      det(0.863, 0.474, 0.103, 0.197, 0.52, "person"),
      det(0.920, 0.449, 0.080, 0.235, 0.51, "person"),
    ];
    let people = vec!["person".to_string()];
    assert_eq!(merge_detections(input.clone(), 0.3, 0.0).len(), 1);
    assert_eq!(
      merge_detections_contained(input, 0.3, 0.0, &people, 0.85).len(),
      2
    );
  }

  #[test]
  fn contained_label_still_merges_a_fragment() {
    // hof: the legs of a person inside the full box
    let input = vec![
      det(0.549, 0.046, 0.080, 0.333, 0.80, "person"),
      det(0.549, 0.233, 0.066, 0.149, 0.60, "person"),
    ];
    let people = vec!["person".to_string()];
    let out = merge_detections_contained(input, 0.3, 0.0, &people, 0.85);
    assert_eq!(out.len(), 1);
    assert!((out[0].confidence - 0.80).abs() < 1e-6);
  }

  #[test]
  fn contained_label_merges_a_small_fragment_cut_at_a_seam() {
    // hof: the legs below a window edge and a crouching child's head, both
    // far below iou 0.3 but on the body's bottom or top edge
    let people = vec!["person".to_string()];
    let legs = vec![
      det(0.570, 0.147, 0.062, 0.213, 0.90, "person"),
      det(0.573, 0.289, 0.034, 0.071, 0.77, "person"),
    ];
    let head = vec![
      det(0.430, 0.773, 0.102, 0.226, 0.80, "person"),
      det(0.433, 0.775, 0.071, 0.076, 0.44, "person"),
    ];
    assert_eq!(
      merge_detections_contained(legs.clone(), 0.3, 0.0, &[], 0.0).len(),
      2
    );
    assert_eq!(
      merge_detections_contained(legs, 0.3, 0.0, &people, 0.85).len(),
      1
    );
    assert_eq!(
      merge_detections_contained(head, 0.3, 0.0, &people, 0.85).len(),
      1
    );
  }

  #[test]
  fn contained_label_keeps_a_person_hidden_inside_another_box() {
    // walkout: a woman on the sofa behind the man, 94 % inside his box
    let input = vec![
      det(0.871, 0.431, 0.129, 0.248, 0.70, "person"),
      det(0.868, 0.524, 0.052, 0.120, 0.53, "person"),
    ];
    let people = vec!["person".to_string()];
    assert_eq!(
      merge_detections_contained(input, 0.3, 0.0, &people, 0.85).len(),
      2
    );
  }

  #[test]
  fn other_labels_ignore_the_containment_rule() {
    let input = vec![
      det(0.863, 0.474, 0.103, 0.197, 0.52, "vehicle"),
      det(0.920, 0.449, 0.080, 0.235, 0.51, "vehicle"),
    ];
    let people = vec!["person".to_string()];
    assert_eq!(
      merge_detections_contained(input, 0.3, 0.0, &people, 0.85).len(),
      1
    );
  }
}
