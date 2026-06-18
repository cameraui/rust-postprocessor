//! IoU helpers. Boxes are normalized `[x, y, width, height]` in `[0.0, 1.0]`.

#[inline]
pub fn box_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
  let ax2 = a[0] + a[2];
  let ay2 = a[1] + a[3];
  let bx2 = b[0] + b[2];
  let by2 = b[1] + b[3];

  let ix1 = a[0].max(b[0]);
  let iy1 = a[1].max(b[1]);
  let ix2 = ax2.min(bx2);
  let iy2 = ay2.min(by2);

  if ix2 <= ix1 || iy2 <= iy1 {
    return 0.0;
  }

  let inter = (ix2 - ix1) * (iy2 - iy1);
  let a_area = a[2] * a[3];
  let b_area = b[2] * b[3];
  let union = a_area + b_area - inter;

  if union > 0.0 {
    inter / union
  } else {
    0.0
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn perfect_overlap() {
    let a = [0.1, 0.1, 0.2, 0.2];
    let b = [0.1, 0.1, 0.2, 0.2];
    assert!((box_iou(&a, &b) - 1.0).abs() < 1e-6);
  }

  #[test]
  fn no_overlap() {
    let a = [0.0, 0.0, 0.1, 0.1];
    let b = [0.5, 0.5, 0.1, 0.1];
    assert_eq!(box_iou(&a, &b), 0.0);
  }

  #[test]
  fn half_overlap_horizontal() {
    let a = [0.0, 0.0, 0.2, 0.2];
    let b = [0.1, 0.0, 0.2, 0.2];
    let v = box_iou(&a, &b);
    assert!((v - 1.0 / 3.0).abs() < 1e-6, "got {v}");
  }

  #[test]
  fn contained() {
    let a = [0.0, 0.0, 1.0, 1.0];
    let b = [0.25, 0.25, 0.5, 0.5];
    let v = box_iou(&a, &b);
    assert!((v - 0.25).abs() < 1e-6, "got {v}");
  }

  #[test]
  fn zero_area() {
    let a = [0.0, 0.0, 0.0, 0.0];
    let b = [0.0, 0.0, 0.5, 0.5];
    assert_eq!(box_iou(&a, &b), 0.0);
  }
}
