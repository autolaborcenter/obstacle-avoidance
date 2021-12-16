use super::cross_numeric;
use crate::{vector, Point2, Vector2};

/// 扩张凸折线（在线计算）
pub(super) struct Enlarger<'a> {
    v: &'a [Point2<f32>],
    len: f32,
    i: usize,
    buffer: Vec<Point2<f32>>,
}

impl<'a> Enlarger<'a> {
    #[inline]
    pub fn new(points: &'a [Point2<f32>], len: f32) -> Self {
        Self {
            v: points,
            len,
            i: 0,
            buffer: vec![],
        }
    }
}

impl Iterator for Enlarger<'_> {
    type Item = Point2<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        // 尖角插值
        if let Some(p) = self.buffer.pop() {
            Some(p)
        } else if self.i < self.v.len() {
            let triple = match self.i {
                0 => [
                    self.v[self.v.len() - 2],
                    self.v[self.v.len() - 1],
                    self.v[0],
                ],
                1 => [self.v[self.v.len() - 1], self.v[0], self.v[1]],
                i => [self.v[i - 2], self.v[i - 1], self.v[i]],
            };
            self.i += 1;
            let c = triple[1];
            let d0 = (triple[0] - c).normalize();
            let d1 = (triple[2] - c).normalize();
            let cross = cross_numeric(d0, d1);
            // 锐角切成直角
            if cross <= 0.0 && d0.dot(&d1).is_sign_positive() {
                // 插入的点逆序保存到缓存
                self.buffer.push(c + self.len * normal(d1));
                self.buffer.push(c - self.len * (d0 + d1).normalize());
                Some(c - self.len * normal(d0))
            } else {
                Some(c + (d0 + d1) * self.len / cross)
            }
        } else {
            None
        }
    }
}

/// 法向量
#[inline]
fn normal(v: Vector2<f32>) -> Vector2<f32> {
    vector(-v[1], v[0])
}
