use crate::config::CENTER;

#[inline]
pub fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

#[inline]
pub fn point_to_segment_distance(p: [f64; 2], v: [f64; 2], w: [f64; 2]) -> f64 {
    let dx = w[0] - v[0];
    let dy = w[1] - v[1];
    let l2 = dx * dx + dy * dy;
    if l2 == 0.0 {
        return distance(p, v);
    }
    let t = (((p[0] - v[0]) * dx + (p[1] - v[1]) * dy) / l2).clamp(0.0, 1.0);
    let proj = [v[0] + t * dx, v[1] + t * dy];
    distance(p, proj)
}

#[inline]
pub fn rotate_about_center(x: f64, y: f64, angle: f64) -> [f64; 2] {
    let dx = x - CENTER;
    let dy = y - CENTER;
    let c = angle.cos();
    let s = angle.sin();
    [CENTER + dx * c - dy * s, CENTER + dx * s + dy * c]
}

#[inline]
pub fn orbital_radius(x: f64, y: f64) -> f64 {
    distance([x, y], [CENTER, CENTER])
}

#[inline]
pub fn angle_between(a: [f64; 2], b: [f64; 2]) -> f64 {
    (b[1] - a[1]).atan2(b[0] - a[0])
}

#[inline]
pub fn line_crosses_circle(start: [f64; 2], end: [f64; 2], center: [f64; 2], radius: f64) -> bool {
    point_to_segment_distance(center, start, end) < radius
}
