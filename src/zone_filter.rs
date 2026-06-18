//! Detection zone filter. Zones are stored normalized to `[0.0, 1.0]`.

use std::collections::HashSet;

use crate::types::Detection;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneMatchType {
  Intersect,
  /// All four corners of the box must be inside the zone.
  Contain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneFilterMode {
  Include,
  Exclude,
}

#[derive(Debug, Clone)]
pub struct ZoneInput {
  /// Empty means "all labels".
  pub labels: Vec<String>,
  pub filter: ZoneFilterMode,
  pub match_type: ZoneMatchType,
  pub is_privacy_mask: bool,
  /// Polygon vertices in `[0, 100]` UI coordinates, not necessarily closed.
  pub points: Vec<[f64; 2]>,
}

#[derive(Debug, Clone)]
pub struct PreparedZone {
  pub labels: HashSet<String>,
  pub filter: ZoneFilterMode,
  pub match_type: ZoneMatchType,
  /// Closed polygon in normalized `[0.0, 1.0]` coordinates.
  pub points: Vec<[f32; 2]>,
}

#[derive(Debug, Clone, Default)]
pub struct PreparedZones {
  pub privacy_masks: Vec<PreparedZone>,
  pub active_zones: Vec<PreparedZone>,
  /// Union of every active zone's allowed labels (lowercased). Empty means
  /// no per-zone label restriction is in effect.
  pub all_labels: HashSet<String>,
}

pub fn prepare_zones(zones: &[ZoneInput]) -> PreparedZones {
  let mut privacy_masks = Vec::new();
  let mut active_zones = Vec::new();
  let mut all_labels: HashSet<String> = HashSet::new();

  for zone in zones {
    let mut points: Vec<[f32; 2]> = zone
      .points
      .iter()
      .map(|p| [(p[0] / 100.0) as f32, (p[1] / 100.0) as f32])
      .collect();

    if points.len() > 1 {
      let first = points[0];
      let last = points[points.len() - 1];
      if (first[0] - last[0]).abs() > 1e-9 || (first[1] - last[1]).abs() > 1e-9 {
        points.push(first);
      }
    }

    let mut labels: HashSet<String> = HashSet::new();
    for label in &zone.labels {
      let lc = label.to_lowercase();
      labels.insert(lc.clone());
      if !zone.is_privacy_mask {
        all_labels.insert(lc);
      }
    }

    let prepared = PreparedZone {
      labels,
      filter: zone.filter,
      match_type: zone.match_type,
      points,
    };

    if zone.is_privacy_mask {
      privacy_masks.push(prepared);
    } else {
      active_zones.push(prepared);
    }
  }

  PreparedZones {
    privacy_masks,
    active_zones,
    all_labels,
  }
}

/// Ray-cast point-in-polygon, with an extra check for points lying exactly
/// on an edge. Polygon must be closed.
fn is_point_in_polygon(px: f32, py: f32, polygon: &[[f32; 2]]) -> bool {
  if polygon.len() < 3 {
    return false;
  }
  let mut inside = false;
  let n = polygon.len();
  let mut j = n - 1;
  for i in 0..n {
    let xi = polygon[i][0];
    let yi = polygon[i][1];
    let xj = polygon[j][0];
    let yj = polygon[j][1];

    // Point exactly on (or very near) the edge.
    let min_x = xi.min(xj);
    let max_x = xi.max(xj);
    let min_y = yi.min(yj);
    let max_y = yi.max(yj);
    if px >= min_x && px <= max_x && py >= min_y && py <= max_y {
      if (xi - xj).abs() < 1e-9 {
        if (px - xi).abs() < 1e-9 {
          return true;
        }
      } else {
        let m = (yj - yi) / (xj - xi);
        if (py - (m * px + (yi - m * xi))).abs() < 1e-9 {
          return true;
        }
      }
    }

    let yi_above = yi > py;
    let yj_above = yj > py;
    if yi_above != yj_above {
      let x_intersect = (xj - xi) * (py - yi) / (yj - yi) + xi;
      if px < x_intersect {
        inside = !inside;
      }
    }

    j = i;
  }
  inside
}

fn do_lines_intersect(a1: [f32; 2], a2: [f32; 2], b1: [f32; 2], b2: [f32; 2]) -> bool {
  let denom = (b2[1] - b1[1]) * (a2[0] - a1[0]) - (b2[0] - b1[0]) * (a2[1] - a1[1]);
  if denom.abs() < 1e-12 {
    return false;
  }
  let ua = ((b2[0] - b1[0]) * (a1[1] - b1[1]) - (b2[1] - b1[1]) * (a1[0] - b1[0])) / denom;
  let ub = ((a2[0] - a1[0]) * (a1[1] - b1[1]) - (a2[1] - a1[1]) * (a1[0] - b1[0])) / denom;
  (0.0..=1.0).contains(&ua) && (0.0..=1.0).contains(&ub)
}

#[inline]
fn box_corners(det: &Detection) -> [[f32; 2]; 4] {
  let x2 = det.x + det.width;
  let y2 = det.y + det.height;
  [[det.x, det.y], [x2, det.y], [x2, y2], [det.x, y2]]
}

fn box_intersects_polygon(det: &Detection, polygon: &[[f32; 2]]) -> bool {
  let corners = box_corners(det);
  let x2 = det.x + det.width;
  let y2 = det.y + det.height;

  for &[cx, cy] in &corners {
    if is_point_in_polygon(cx, cy, polygon) {
      return true;
    }
  }
  for &[px, py] in polygon {
    if px >= det.x && px <= x2 && py >= det.y && py <= y2 {
      return true;
    }
  }
  let edges = [
    (corners[0], corners[1]),
    (corners[1], corners[2]),
    (corners[2], corners[3]),
    (corners[3], corners[0]),
  ];
  if polygon.len() < 2 {
    return false;
  }
  for i in 0..(polygon.len() - 1) {
    for &(ea, eb) in &edges {
      if do_lines_intersect(ea, eb, polygon[i], polygon[i + 1]) {
        return true;
      }
    }
  }
  false
}

fn box_contained_in_polygon(det: &Detection, polygon: &[[f32; 2]]) -> bool {
  for &[cx, cy] in &box_corners(det) {
    if !is_point_in_polygon(cx, cy, polygon) {
      return false;
    }
  }
  true
}

#[inline]
fn zone_accepts_label(zone: &PreparedZone, lc_label: &str) -> bool {
  zone.labels.is_empty() || zone.labels.contains(lc_label)
}

/// Returns the indices of detections that pass the confidence + zone filter.
pub fn filter_indices(
  detections: &[Detection],
  zones: &PreparedZones,
  min_confidence: f32,
) -> Vec<u32> {
  let PreparedZones {
    privacy_masks,
    active_zones,
    all_labels,
  } = zones;

  let mut out: Vec<u32> = Vec::with_capacity(detections.len());
  for (i, det) in detections.iter().enumerate() {
    if det.confidence < min_confidence {
      continue;
    }

    if active_zones.is_empty() && privacy_masks.is_empty() {
      out.push(i as u32);
      continue;
    }

    let label_lc = det.label.to_lowercase();
    if !all_labels.is_empty() && !all_labels.contains(&label_lc) {
      continue;
    }

    let mut dropped = false;
    for mask in privacy_masks {
      if box_contained_in_polygon(det, &mask.points) {
        dropped = true;
        break;
      }
    }
    if dropped {
      continue;
    }

    let mut has_include_zone = false;
    let mut satisfies_include = false;

    for zone in active_zones {
      if !zone_accepts_label(zone, &label_lc) {
        continue;
      }
      let intersects = box_intersects_polygon(det, &zone.points);
      let contained = intersects && box_contained_in_polygon(det, &zone.points);

      match zone.filter {
        ZoneFilterMode::Exclude => match zone.match_type {
          ZoneMatchType::Contain => {
            if intersects {
              dropped = true;
              break;
            }
          }
          ZoneMatchType::Intersect => {
            if intersects || contained {
              dropped = true;
              break;
            }
          }
        },
        ZoneFilterMode::Include => {
          has_include_zone = true;
          if !satisfies_include {
            match zone.match_type {
              ZoneMatchType::Contain => {
                if contained {
                  satisfies_include = true;
                }
              }
              ZoneMatchType::Intersect => {
                if intersects {
                  satisfies_include = true;
                }
              }
            }
          }
        }
      }
    }

    if dropped {
      continue;
    }
    if has_include_zone && !satisfies_include {
      continue;
    }
    out.push(i as u32);
  }
  out
}

#[cfg(test)]
fn filter_detections(
  detections: Vec<Detection>,
  zones: &PreparedZones,
  min_confidence: f32,
) -> Vec<Detection> {
  let indices = filter_indices(&detections, zones, min_confidence);
  if indices.len() == detections.len() {
    return detections;
  }
  let mut slots: Vec<Option<Detection>> = detections.into_iter().map(Some).collect();
  indices
    .iter()
    .filter_map(|&i| slots[i as usize].take())
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;

  fn det(x: f32, y: f32, w: f32, h: f32, label: &str) -> Detection {
    Detection {
      x,
      y,
      width: w,
      height: h,
      confidence: 0.9,
      label: label.to_string(),
    }
  }

  fn rect_zone(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    mode: ZoneFilterMode,
    mt: ZoneMatchType,
    labels: Vec<String>,
  ) -> ZoneInput {
    ZoneInput {
      labels,
      filter: mode,
      match_type: mt,
      is_privacy_mask: false,
      points: vec![[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
    }
  }

  #[test]
  fn confidence_filter_only() {
    let zones = prepare_zones(&[]);
    let mut a = det(0.1, 0.1, 0.2, 0.2, "person");
    a.confidence = 0.4;
    let mut b = det(0.3, 0.3, 0.2, 0.2, "person");
    b.confidence = 0.8;
    let out = filter_detections(vec![a, b], &zones, 0.5);
    assert_eq!(out.len(), 1);
    assert!(out[0].confidence >= 0.5);
  }

  #[test]
  fn include_zone_contains_box() {
    let zones = prepare_zones(&[rect_zone(
      25.0,
      25.0,
      75.0,
      75.0,
      ZoneFilterMode::Include,
      ZoneMatchType::Contain,
      vec![],
    )]);
    let inside = det(0.30, 0.30, 0.20, 0.20, "person");
    let outside = det(0.85, 0.85, 0.10, 0.10, "person");
    let out = filter_detections(vec![inside, outside], &zones, 0.0);
    assert_eq!(out.len(), 1);
  }

  #[test]
  fn include_intersect_keeps_partial_overlap() {
    let zones = prepare_zones(&[rect_zone(
      25.0,
      25.0,
      75.0,
      75.0,
      ZoneFilterMode::Include,
      ZoneMatchType::Intersect,
      vec![],
    )]);
    let partial = det(0.70, 0.30, 0.20, 0.20, "person");
    let out = filter_detections(vec![partial], &zones, 0.0);
    assert_eq!(out.len(), 1);
  }

  #[test]
  fn exclude_zone_drops_box() {
    let zones = prepare_zones(&[rect_zone(
      25.0,
      25.0,
      75.0,
      75.0,
      ZoneFilterMode::Exclude,
      ZoneMatchType::Intersect,
      vec![],
    )]);
    let inside = det(0.30, 0.30, 0.20, 0.20, "person");
    let outside = det(0.05, 0.05, 0.10, 0.10, "person");
    let out = filter_detections(vec![inside, outside], &zones, 0.0);
    assert_eq!(out.len(), 1);
    assert!((out[0].x - 0.05).abs() < 1e-6);
  }

  #[test]
  fn privacy_mask_drops_contained_box() {
    let mut mask = rect_zone(
      0.0,
      0.0,
      50.0,
      50.0,
      ZoneFilterMode::Include,
      ZoneMatchType::Intersect,
      vec![],
    );
    mask.is_privacy_mask = true;
    let zones = prepare_zones(&[mask]);
    let inside = det(0.10, 0.10, 0.20, 0.20, "person");
    let outside = det(0.60, 0.60, 0.20, 0.20, "person");
    let out = filter_detections(vec![inside, outside], &zones, 0.0);
    assert_eq!(out.len(), 1);
    assert!((out[0].x - 0.60).abs() < 1e-6);
  }

  #[test]
  fn label_filter_restricts_globally() {
    // Zone for cars only: persons dropped even outside it (union of zone
    // labels is a global allow-list).
    let zones = prepare_zones(&[rect_zone(
      0.0,
      0.0,
      100.0,
      100.0,
      ZoneFilterMode::Include,
      ZoneMatchType::Intersect,
      vec!["car".to_string()],
    )]);
    let person = det(0.30, 0.30, 0.20, 0.20, "person");
    let car = det(0.30, 0.30, 0.20, 0.20, "car");
    let out = filter_detections(vec![person, car], &zones, 0.0);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].label, "car");
  }

  #[test]
  fn label_zone_with_other_label_zone_combined() {
    let zones = prepare_zones(&[
      rect_zone(
        0.0,
        0.0,
        100.0,
        100.0,
        ZoneFilterMode::Include,
        ZoneMatchType::Intersect,
        vec!["person".to_string()],
      ),
      rect_zone(
        0.0,
        0.0,
        100.0,
        100.0,
        ZoneFilterMode::Include,
        ZoneMatchType::Intersect,
        vec!["car".to_string()],
      ),
    ]);
    let person = det(0.30, 0.30, 0.20, 0.20, "person");
    let car = det(0.30, 0.30, 0.20, 0.20, "car");
    let cat = det(0.30, 0.30, 0.20, 0.20, "cat");
    let out = filter_detections(vec![person, car, cat], &zones, 0.0);
    assert_eq!(out.len(), 2);
  }

  fn hof_zone() -> ZoneInput {
    ZoneInput {
      labels: vec![
        "motion".to_string(),
        "person".to_string(),
        "vehicle".to_string(),
        "animal".to_string(),
      ],
      filter: ZoneFilterMode::Include,
      match_type: ZoneMatchType::Contain,
      is_privacy_mask: false,
      points: vec![
        [0.0, 39.0],
        [7.0, 25.0],
        [50.0, 22.0],
        [50.0, 33.0],
        [63.0, 34.0],
        [64.0, 19.0],
        [75.0, 20.0],
        [82.0, 13.0],
        [92.0, 42.0],
        [100.0, 80.0],
        [100.0, 100.0],
        [0.0, 100.0],
        [0.0, 70.0],
      ],
    }
  }

  #[test]
  fn hof_zone_street_motion_filtered() {
    let zones = prepare_zones(&[hof_zone()]);
    let street_car = det(0.35, 0.05, 0.15, 0.10, "motion");
    let out = filter_detections(vec![street_car], &zones, 0.0);
    assert_eq!(out.len(), 0, "street motion should be filtered out");
  }

  #[test]
  fn hof_zone_courtyard_motion_passes() {
    let zones = prepare_zones(&[hof_zone()]);
    let courtyard = det(0.30, 0.60, 0.15, 0.15, "motion");
    let out = filter_detections(vec![courtyard], &zones, 0.0);
    assert_eq!(out.len(), 1, "courtyard motion should pass");
  }

  #[test]
  fn hof_zone_boundary_straddling_filtered() {
    let zones = prepare_zones(&[hof_zone()]);
    // At x≈0.50 the zone boundary is at y≈0.22; the top-left corner is outside.
    let straddling = det(0.45, 0.15, 0.10, 0.20, "motion");
    let out = filter_detections(vec![straddling], &zones, 0.0);
    assert_eq!(
      out.len(),
      0,
      "boundary-straddling motion should be filtered (Contain)"
    );
  }

  #[test]
  fn hof_zone_near_top_boundary_inside() {
    // Edge (75,20)→(82,13) interpolates to y≈0.17 at x=0.78; the bbox sits
    // fully below it so all four corners are inside.
    let zones = prepare_zones(&[hof_zone()]);
    let near_boundary = det(0.78, 0.20, 0.04, 0.07, "motion");
    let out = filter_detections(vec![near_boundary], &zones, 0.0);
    assert_eq!(
      out.len(),
      1,
      "motion just inside the zone boundary should pass"
    );
  }

  #[test]
  fn hof_zone_left_edge_outside() {
    // Left boundary at x=0 starts at y=0.39; this bbox's top is at y=0.30.
    let zones = prepare_zones(&[hof_zone()]);
    let left_top = det(0.0, 0.30, 0.05, 0.05, "motion");
    let out = filter_detections(vec![left_top], &zones, 0.0);
    assert_eq!(
      out.len(),
      0,
      "motion above left zone boundary should be filtered"
    );
  }

  #[test]
  fn hof_zone_mixed_street_and_courtyard() {
    let zones = prepare_zones(&[hof_zone()]);
    let street = det(0.20, 0.02, 0.12, 0.08, "motion");
    let courtyard = det(0.40, 0.70, 0.10, 0.10, "motion");
    let straddling = det(0.50, 0.18, 0.10, 0.15, "motion");
    let out = filter_detections(vec![street, courtyard, straddling], &zones, 0.0);
    assert_eq!(out.len(), 1, "only courtyard motion should pass");
    assert!(
      (out[0].y - 0.70).abs() < 1e-6,
      "the surviving detection should be the courtyard one"
    );
  }

  #[test]
  fn hof_zone_person_on_street_filtered() {
    let zones = prepare_zones(&[hof_zone()]);
    let person_street = det(0.40, 0.05, 0.05, 0.12, "person");
    let person_yard = det(0.40, 0.50, 0.05, 0.12, "person");
    let out = filter_detections(vec![person_street, person_yard], &zones, 0.0);
    assert_eq!(out.len(), 1, "person on street should be filtered");
    assert!((out[0].y - 0.50).abs() < 1e-6);
  }

  #[test]
  fn hof_zone_large_motion_bbox_filtered() {
    // A full-frame motion bbox (e.g. a lighting change) is still filtered
    // under Contain because its top corners are outside the zone.
    let zones = prepare_zones(&[hof_zone()]);
    let large = det(0.0, 0.0, 1.0, 1.0, "motion");
    let out = filter_detections(vec![large], &zones, 0.0);
    assert_eq!(
      out.len(),
      0,
      "full-frame motion bbox should be filtered (top corners outside)"
    );
  }

  #[test]
  fn auto_close_polygon() {
    let zone = ZoneInput {
      labels: vec![],
      filter: ZoneFilterMode::Include,
      match_type: ZoneMatchType::Intersect,
      is_privacy_mask: false,
      points: vec![[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
    };
    let zones = prepare_zones(&[zone]);
    assert_eq!(zones.active_zones[0].points.len(), 5);
    let first = zones.active_zones[0].points[0];
    let last = zones.active_zones[0].points[4];
    assert!((first[0] - last[0]).abs() < 1e-6);
    assert!((first[1] - last[1]).abs() < 1e-6);
  }
}
