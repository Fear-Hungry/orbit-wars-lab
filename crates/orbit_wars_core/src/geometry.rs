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

/// Exact port of the official `swept_pair_hit`: true iff a fleet moving `a->b`
/// and a planet moving `p0->p1` come within `r` of each other for some t in
/// [0, 1]. Both segments are treated as linear over the tick (planet rotation
/// linearised to its chord). Accounting for the planet's motion is what keeps a
/// rotating planet from registering a phantom hit at the position it has left.
#[inline]
pub fn swept_pair_hit(a: [f64; 2], b: [f64; 2], p0: [f64; 2], p1: [f64; 2], r: f64) -> bool {
    let d0x = a[0] - p0[0];
    let d0y = a[1] - p0[1];
    let dvx = (b[0] - a[0]) - (p1[0] - p0[0]);
    let dvy = (b[1] - a[1]) - (p1[1] - p0[1]);
    let aa = dvx * dvx + dvy * dvy;
    let bb = 2.0 * (d0x * dvx + d0y * dvy);
    let cc = d0x * d0x + d0y * d0y - r * r;
    if aa < 1e-12 {
        return cc <= 0.0;
    }
    let disc = bb * bb - 4.0 * aa * cc;
    if disc < 0.0 {
        return false;
    }
    let sq = disc.sqrt();
    let t1 = (-bb - sq) / (2.0 * aa);
    let t2 = (-bb + sq) / (2.0 * aa);
    t2 >= 0.0 && t1 <= 1.0
}
