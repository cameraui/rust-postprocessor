//! Line-crossing detection. The crossing line is perpendicular to the handle
//! segment the user draws; crossings are detected by the cross-product sign of
//! a track's prev->curr segment against the line.

use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineDirectionFilter {
  Both,
  AToB,
  BToA,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossingDirection {
  AToB,
  BToA,
}

#[derive(Debug, Clone)]
pub struct DetectionLineInput {
  pub name: String,
  pub direction: LineDirectionFilter,
  /// Allowed labels (case-insensitive). Empty means "all labels".
  pub labels: Vec<String>,
  /// Handle endpoints in [0, 100] UI coordinates.
  pub points: [[f64; 2]; 2],
}

#[derive(Debug, Clone)]
pub struct PreparedLine {
  pub name: String,
  pub direction: LineDirectionFilter,
  /// Lowercased labels; empty set means all allowed.
  pub labels: HashSet<String>,
  pub line_a: [f32; 2],
  pub line_b: [f32; 2],
}

#[derive(Debug, Clone)]
pub struct LineCrossingEvent {
  pub line_name: String,
  pub direction: CrossingDirection,
  pub track_id: u32,
  pub label: String,
  pub confidence: f32,
  pub timestamp_ms: f64,
  pub prev_pos: [f32; 2],
  pub curr_pos: [f32; 2],
}

/// Build prepared lines from UI input. `aspect_ratio` is the camera `width / height`.
pub fn prepare_lines(lines: &[DetectionLineInput], aspect_ratio: f32) -> Vec<PreparedLine> {
  lines
    .iter()
    .map(|line| {
      let h1x = (line.points[0][0] / 100.0) as f32;
      let h1y = (line.points[0][1] / 100.0) as f32;
      let h2x = (line.points[1][0] / 100.0) as f32;
      let h2y = (line.points[1][1] / 100.0) as f32;

      let mid_x = (h1x + h2x) * 0.5;
      let mid_y = (h1y + h2y) * 0.5;

      // Perpendicular is computed in visual (aspect-corrected) space, then
      // scaled back to normalized space so it looks perpendicular to the user.
      let dx_vis = (h2x - h1x) * aspect_ratio;
      let dy_vis = h2y - h1y;
      let perp_x_vis = -dy_vis;
      let perp_y_vis = dx_vis;

      let perp_x_norm = perp_x_vis / aspect_ratio;
      let perp_y_norm = perp_y_vis;
      let perp_len = (perp_x_norm * perp_x_norm + perp_y_norm * perp_y_norm)
        .sqrt()
        .max(1e-12);
      let handle_len = ((h2x - h1x).powi(2) + (h2y - h1y).powi(2))
        .sqrt()
        .max(1e-12);
      let scale = handle_len / perp_len;
      let perp_x = perp_x_norm * scale;
      let perp_y = perp_y_norm * scale;

      let line_a = [mid_x - perp_x * 0.5, mid_y - perp_y * 0.5];
      let line_b = [mid_x + perp_x * 0.5, mid_y + perp_y * 0.5];

      let labels: HashSet<String> = line.labels.iter().map(|l| l.to_lowercase()).collect();

      PreparedLine {
        name: line.name.clone(),
        direction: line.direction,
        labels,
        line_a,
        line_b,
      }
    })
    .collect()
}

/// Intersect track segment `(a,b)` with line `(c,d)`. Returns the signed cross
/// product (positive A->B, negative B->A), or 0 if disjoint or parallel.
#[inline]
pub fn segment_intersection(
  ax: f32,
  ay: f32,
  bx: f32,
  by: f32,
  cx: f32,
  cy: f32,
  dx: f32,
  dy: f32,
) -> f32 {
  let denom = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx);
  if denom.abs() < 1e-12 {
    return 0.0;
  }

  let t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denom;
  let u = ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / denom;

  if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
    let track_dx = bx - ax;
    let track_dy = by - ay;
    let line_dx = dx - cx;
    let line_dy = dy - cy;
    track_dx * line_dy - track_dy * line_dx
  } else {
    0.0
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn line(name: &str, p1: [f64; 2], p2: [f64; 2]) -> DetectionLineInput {
    DetectionLineInput {
      name: name.to_string(),
      direction: LineDirectionFilter::Both,
      labels: Vec::new(),
      points: [p1, p2],
    }
  }

  #[test]
  fn prepare_horizontal_handle_yields_vertical_line() {
    let prepared = prepare_lines(&[line("h", [10.0, 50.0], [90.0, 50.0])], 16.0 / 9.0);
    assert_eq!(prepared.len(), 1);
    let p = &prepared[0];
    assert!((p.line_a[0] - p.line_b[0]).abs() < 1e-4);
    assert!((p.line_a[0] - 0.5).abs() < 1e-4);
  }

  #[test]
  fn segment_intersection_basic() {
    let cross = segment_intersection(0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0);
    assert!(cross != 0.0);
  }

  #[test]
  fn segment_intersection_no_overlap() {
    let cross = segment_intersection(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    assert_eq!(cross, 0.0);
  }

  #[test]
  fn segment_intersection_disjoint() {
    let cross = segment_intersection(0.0, 0.0, 0.1, 0.1, 0.5, 0.5, 0.6, 0.6);
    assert_eq!(cross, 0.0);
  }

  #[test]
  fn cross_sign_indicates_direction() {
    let cross_lr = segment_intersection(0.4, 0.5, 0.6, 0.5, 0.5, 0.0, 0.5, 1.0);
    assert!(cross_lr > 0.0);
    let cross_rl = segment_intersection(0.6, 0.5, 0.4, 0.5, 0.5, 0.0, 0.5, 1.0);
    assert!(cross_rl < 0.0);
  }
}
