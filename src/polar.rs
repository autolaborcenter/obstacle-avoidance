use crate::{vector, Point2};

#[derive(Clone, Copy, Debug)]
pub struct Polar {
    pub rho: f32,
    pub theta: f32,
}

impl From<Point2<f32>> for Polar {
    #[inline]
    fn from(p: Point2<f32>) -> Self {
        Self {
            rho: p.coords.norm(),
            theta: p.coords[1].atan2(p.coords[0]),
        }
    }
}

impl From<Polar> for Point2<f32> {
    #[inline]
    fn from(p: Polar) -> Self {
        let (sin, cos) = p.theta.sin_cos();
        Point2 {
            coords: vector(cos, sin) * p.rho,
        }
    }
}
