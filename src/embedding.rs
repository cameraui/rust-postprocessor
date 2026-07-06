//! Appearance embedding for ReID — a 64-element feature vector per crop:
//!   [0..24] HSV histogram, [24..40] gradient orientations,
//!   [40..56] 4×4 spatial grid, [60] brightness, [61] aspect ratio (rest 0).

use wide::f32x8;

const MIN_CROP_W: u32 = 20;
const MIN_CROP_H: u32 = 40;

/// Below this brightness, color features get reduced weight.
const DARK_THRESHOLD: f64 = 0.3;

pub const EMBEDDING_SIZE: usize = 64;

/// Compute a 64-element appearance embedding from an RGB24 region. `bbox` is
/// normalized `[x, y, w, h]`. Returns `None` if the crop is too small.
pub fn compute_embedding(
  pixels: &[u8],
  img_w: u32,
  img_h: u32,
  bbox: [f32; 4],
) -> Option<Vec<f64>> {
  let [bx, by, bw, bh] = bbox;

  let x0 = ((bx * img_w as f32) as u32).min(img_w.saturating_sub(1));
  let y0 = ((by * img_h as f32) as u32).min(img_h.saturating_sub(1));
  let x1 = (((bx + bw) * img_w as f32).ceil() as u32).min(img_w);
  let y1 = (((by + bh) * img_h as f32).ceil() as u32).min(img_h);

  let crop_w = x1.saturating_sub(x0);
  let crop_h = y1.saturating_sub(y0);

  if crop_w < MIN_CROP_W || crop_h < MIN_CROP_H {
    return None;
  }

  let mut emb = vec![0.0f64; EMBEDDING_SIZE];

  accumulate_features(pixels, img_w, x0, y0, crop_w, crop_h, &mut emb);
  emb[61] = crop_h as f64 / crop_w as f64;

  // Normalize histogram segments independently for scale invariance.
  l2_normalize(&mut emb[0..24]);
  l2_normalize(&mut emb[24..40]);
  l2_normalize(&mut emb[40..56]);

  Some(emb)
}

/// Fill the color histogram `[0..24]`, gradient histogram `[24..40]`, spatial
/// grid `[40..56]` and mean brightness `[60]` in a single pass over the crop,
/// so each pixel row is pulled through the cache once instead of four times.
fn accumulate_features(
  pixels: &[u8],
  img_w: u32,
  x0: u32,
  y0: u32,
  crop_w: u32,
  crop_h: u32,
  emb: &mut [f64],
) {
  let stride = (img_w * 3) as usize;
  let mag_threshold = 30.0f32;
  // Gradients need neighbors on both axes.
  let gradients = crop_w >= 3 && crop_h >= 3;

  let mut grid_counts = [0u32; 16];
  let mut brightness_sum = 0u64;
  let mut brightness_count = 0u64;

  for dy in 0..crop_h {
    let y = (y0 + dy) as usize;
    let row_start = y * stride + (x0 as usize) * 3;
    let cy = ((dy * 4) / crop_h).min(3) as usize;
    let grad_row = gradients && dy >= 1 && dy < crop_h - 1;

    for dx in 0..crop_w {
      let off = row_start + (dx as usize) * 3;
      if off + 2 >= pixels.len() {
        continue;
      }
      let r = pixels[off];
      let g = pixels[off + 1];
      let b = pixels[off + 2];

      let (h, s, v) = rgb_to_hsv(r, g, b);
      let hbin = ((h * 8.0) as usize).min(7);
      emb[hbin] += 1.0;
      let sbin = ((s * 8.0) as usize).min(7);
      emb[8 + sbin] += 1.0;
      let vbin = ((v * 8.0) as usize).min(7);
      emb[16 + vbin] += 1.0;

      let cx = ((dx * 4) / crop_w).min(3) as usize;
      let cell = cy * 4 + cx;
      // Approximate luminance: (R + 2G + B) / 4.
      let lum = (r as u32 + g as u32 * 2 + b as u32) / 4;
      emb[40 + cell] += lum as f64;
      grid_counts[cell] += 1;

      brightness_sum += g as u64;
      brightness_count += 1;

      if grad_row && dx >= 1 && dx < crop_w - 1 {
        let x = (x0 + dx) as usize;
        // Green channel as a luma proxy.
        let idx_l = y * stride + (x - 1) * 3 + 1;
        let idx_r = y * stride + (x + 1) * 3 + 1;
        let idx_u = (y - 1) * stride + x * 3 + 1;
        let idx_d = (y + 1) * stride + x * 3 + 1;
        if idx_d >= pixels.len() || idx_r >= pixels.len() {
          continue;
        }

        let gx = pixels[idx_r] as f32 - pixels[idx_l] as f32;
        let gy = pixels[idx_d] as f32 - pixels[idx_u] as f32;

        let mag = (gx * gx + gy * gy).sqrt();
        if mag < 1.0 {
          continue;
        }

        let angle = gy.atan2(gx) + std::f32::consts::PI; // shift to [0, 2π)
        let bin = (((angle / (2.0 * std::f32::consts::PI)) * 8.0) as usize).min(7);
        let level = if mag > mag_threshold { 8 } else { 0 };
        emb[24 + level + bin] += mag as f64;
      }
    }
  }

  for i in 0..16 {
    if grid_counts[i] > 0 {
      emb[40 + i] /= grid_counts[i] as f64 * 255.0;
    }
  }
  if brightness_count > 0 {
    emb[60] = brightness_sum as f64 / (brightness_count as f64 * 255.0);
  }
}

/// Adaptive weighted cosine distance in `[0.0, 1.0]` (0 = identical). Color
/// weight shrinks in dark scenes (brightness at index 60).
pub fn embedding_distance(a: &[f64], b: &[f64]) -> f64 {
  if a.len() < EMBEDDING_SIZE || b.len() < EMBEDDING_SIZE {
    return f64::INFINITY;
  }

  let brightness_a = a[60];
  let brightness_b = b[60];
  let avg_brightness = (brightness_a + brightness_b) * 0.5;

  let color_weight = if avg_brightness > DARK_THRESHOLD {
    0.5
  } else {
    0.1
  };
  let gradient_weight = 0.3;
  let spatial_weight = 1.0 - color_weight - gradient_weight;

  let color_dist = cosine_distance_simd(&a[0..24], &b[0..24]);
  let gradient_dist = cosine_distance_simd(&a[24..40], &b[24..40]);
  let spatial_dist = cosine_distance_simd(&a[40..56], &b[40..56]);

  let dist =
    color_weight * color_dist + gradient_weight * gradient_dist + spatial_weight * spatial_dist;

  dist.clamp(0.0, 1.0)
}

/// RGB → HSV with all components in `[0, 1]`.
#[inline(always)]
fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
  let rf = r as f32 / 255.0;
  let gf = g as f32 / 255.0;
  let bf = b as f32 / 255.0;

  let max = rf.max(gf).max(bf);
  let min = rf.min(gf).min(bf);
  let delta = max - min;

  let v = max;
  let s = if max > 0.0 { delta / max } else { 0.0 };

  let h = if delta < 1e-6 {
    0.0
  } else if max == rf {
    ((gf - bf) / delta).rem_euclid(6.0) / 6.0
  } else if max == gf {
    ((bf - rf) / delta + 2.0) / 6.0
  } else {
    ((rf - gf) / delta + 4.0) / 6.0
  };

  (h, s, v)
}

/// L2-normalize in place; zero-norm slices are left unchanged.
fn l2_normalize(data: &mut [f64]) {
  let norm_sq: f64 = data.iter().map(|x| x * x).sum();
  if norm_sq < 1e-12 {
    return;
  }
  let inv_norm = 1.0 / norm_sq.sqrt();
  for x in data.iter_mut() {
    *x *= inv_norm;
  }
}

/// Cosine distance (`1.0 - similarity`, in `[0.0, 2.0]`) over f32x8 chunks.
fn cosine_distance_simd(a: &[f64], b: &[f64]) -> f64 {
  let n = a.len().min(b.len());
  if n == 0 {
    return 1.0;
  }

  let chunks = n / 8;
  let remainder = n % 8;

  let mut dot_acc = f32x8::ZERO;
  let mut mag_a_acc = f32x8::ZERO;
  let mut mag_b_acc = f32x8::ZERO;

  for i in 0..chunks {
    let base = i * 8;
    let av = f32x8::new([
      a[base] as f32,
      a[base + 1] as f32,
      a[base + 2] as f32,
      a[base + 3] as f32,
      a[base + 4] as f32,
      a[base + 5] as f32,
      a[base + 6] as f32,
      a[base + 7] as f32,
    ]);
    let bv = f32x8::new([
      b[base] as f32,
      b[base + 1] as f32,
      b[base + 2] as f32,
      b[base + 3] as f32,
      b[base + 4] as f32,
      b[base + 5] as f32,
      b[base + 6] as f32,
      b[base + 7] as f32,
    ]);

    dot_acc += av * bv;
    mag_a_acc += av * av;
    mag_b_acc += bv * bv;
  }

  let dot_arr: [f32; 8] = dot_acc.into();
  let mag_a_arr: [f32; 8] = mag_a_acc.into();
  let mag_b_arr: [f32; 8] = mag_b_acc.into();

  let mut dot: f64 = dot_arr.iter().map(|x| *x as f64).sum();
  let mut mag_a: f64 = mag_a_arr.iter().map(|x| *x as f64).sum();
  let mut mag_b: f64 = mag_b_arr.iter().map(|x| *x as f64).sum();

  let base = chunks * 8;
  for i in 0..remainder {
    let av = a[base + i];
    let bv = b[base + i];
    dot += av * bv;
    mag_a += av * av;
    mag_b += bv * bv;
  }

  let denom = (mag_a * mag_b).sqrt();
  if denom < 1e-12 {
    return 0.0; // both vectors zero → treat as identical
  }

  1.0 - (dot / denom)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn solid_frame(r: u8, g: u8, b: u8) -> (Vec<u8>, u32, u32) {
    let w = 64u32;
    let h = 64u32;
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for i in 0..(w * h) as usize {
      pixels[i * 3] = r;
      pixels[i * 3 + 1] = g;
      pixels[i * 3 + 2] = b;
    }
    (pixels, w, h)
  }

  #[test]
  fn same_frame_zero_distance() {
    let (pixels, w, h) = solid_frame(128, 64, 200);
    let bbox = [0.1, 0.1, 0.8, 0.8];
    let emb = compute_embedding(&pixels, w, h, bbox).unwrap();
    let dist = embedding_distance(&emb, &emb);
    assert!(
      dist < 1e-6,
      "Same embedding should have ~0 distance, got {dist}"
    );
  }

  #[test]
  fn different_colors_high_distance() {
    // Bright colors so adaptive weighting favors color features.
    let (pixels_a, w, h) = solid_frame(255, 200, 100);
    let (pixels_b, _, _) = solid_frame(50, 100, 255);
    let bbox = [0.0, 0.0, 1.0, 1.0];
    let emb_a = compute_embedding(&pixels_a, w, h, bbox).unwrap();
    let emb_b = compute_embedding(&pixels_b, w, h, bbox).unwrap();
    let dist = embedding_distance(&emb_a, &emb_b);
    assert!(
      dist > 0.05,
      "Different colors should have measurable distance, got {dist}"
    );
  }

  #[test]
  fn too_small_crop_returns_none() {
    let (pixels, w, h) = solid_frame(128, 128, 128);
    let bbox = [0.0, 0.0, 10.0 / w as f32, 10.0 / h as f32];
    let result = compute_embedding(&pixels, w, h, bbox);
    assert!(result.is_none());
  }

  #[test]
  fn embedding_has_correct_size() {
    let (pixels, w, h) = solid_frame(100, 150, 200);
    let bbox = [0.0, 0.0, 1.0, 1.0];
    let emb = compute_embedding(&pixels, w, h, bbox).unwrap();
    assert_eq!(emb.len(), EMBEDDING_SIZE);
  }

  #[test]
  fn cosine_distance_identical_is_zero() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let d = cosine_distance_simd(&a, &a);
    assert!(d.abs() < 1e-6);
  }

  #[test]
  fn cosine_distance_orthogonal_is_one() {
    let a = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let d = cosine_distance_simd(&a, &b);
    assert!((d - 1.0).abs() < 1e-6);
  }
}
