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
  pub labels: HashSet<String>,
  pub p1: [f32; 2],
  pub p2: [f32; 2],
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

pub fn prepare_lines(lines: &[DetectionLineInput]) -> Vec<PreparedLine> {
  lines
    .iter()
    .map(|line| {
      let labels: HashSet<String> = line.labels.iter().map(|l| l.to_lowercase()).collect();

      PreparedLine {
        name: line.name.clone(),
        direction: line.direction,
        labels,
        p1: [
          (line.points[0][0] / 100.0) as f32,
          (line.points[0][1] / 100.0) as f32,
        ],
        p2: [
          (line.points[1][0] / 100.0) as f32,
          (line.points[1][1] / 100.0) as f32,
        ],
      }
    })
    .collect()
}

#[inline]
#[allow(clippy::too_many_arguments)]
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
  fn prepare_keeps_the_drawn_line() {
    let prepared = prepare_lines(&[line("h", [10.0, 50.0], [90.0, 60.0])]);
    assert_eq!(prepared.len(), 1);
    let p = &prepared[0];
    assert_eq!(p.p1, [0.1, 0.5]);
    assert_eq!(p.p2, [0.9, 0.6]);
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
