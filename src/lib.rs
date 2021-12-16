use nalgebra::{Isometry2, Point2, UnitComplex, Vector2};
use std::{cmp::Ordering, f32::consts::FRAC_PI_2};

mod builder;
mod enlarger;
mod polar;
mod simplify;

pub use polar::Polar;
pub use simplify::{convex_from_origin, fit};

use builder::ObstacleBuilder;
use enlarger::Enlarger;

/// 障碍物对象
#[derive(Clone)]
pub struct Obstacle {
    pub front: Vec<Polar>,
    pub back: Vec<Polar>,
}

/// 障碍物构造异常
pub enum ObstacleError {
    Empty,                // 障碍物不存在
    Besieged(Vec<Polar>), // 障碍物包围了机器人
}

impl Obstacle {
    /// 构造障碍物对象
    pub fn new(
        sensor_on_robot: Isometry2<f32>,
        points: &[Point2<f32>],
        width: f32,
    ) -> Result<Self, ObstacleError> {
        if points.len() < 3 {
            Err(ObstacleError::Empty)
        } else {
            let mut vertex =
                Enlarger::new(points, width * 0.5).map(|p| Polar::from(sensor_on_robot * p));

            let (first, second) = {
                let head = vertex.next().unwrap();
                let mut builder = ObstacleBuilder::new(head.theta);
                vertex.for_each(|p| builder.push(p));
                builder.push(head);
                builder.collect()
            };

            if first.is_empty() {
                Err(ObstacleError::Besieged(second))
            } else if second.is_empty() {
                Err(ObstacleError::Besieged(first))
            } else if first.last().unwrap().theta > second.last().unwrap().theta {
                Ok(Self {
                    front: first,
                    back: second,
                })
            } else {
                Ok(Self {
                    front: second,
                    back: first,
                })
            }
        }
    }

    /// 找到避开障碍物的方法：路线离开障碍物的位置和目标点方向
    pub fn to_avoid(
        &self,
        slice: impl IntoIterator<Item = Point2<f32>>,
        inside: impl Fn(&Point2<f32>) -> bool,
        trend: &mut f32,
    ) -> (usize, f32) {
        let mut skip = 0;
        let part = slice
            .into_iter()
            .skip_while(|p| {
                if !inside(p) {
                    skip += 1;
                    true
                } else {
                    false
                }
            })
            .take_while(&inside)
            .collect::<Vec<_>>();
        let mut iter = part.iter().copied().enumerate();
        let next = self.go_through(&mut iter, trend);
        (
            match part.len() {
                0 => 0,
                len => skip + iter.next().map_or(len, |(i, _)| i) - 1,
            },
            -next.theta.clamp(-FRAC_PI_2, FRAC_PI_2),
        )
    }

    /// 判断点是否在障碍物内
    #[inline]
    pub fn contains(&self, p: Point2<f32>) -> bool {
        self.check_relation(p.into()) == Some(Ordering::Equal)
    }

    /// 计算线段与障碍物交点，返回可能绕过障碍物的倒序的一列点
    fn go_through(
        &self,
        path: &mut impl Iterator<Item = (usize, Point2<f32>)>,
        trend: &mut f32,
    ) -> Polar {
        // 找到离开位置，如果可直达，直接去
        while let Some((_, p)) = path.next() {
            let polar = p.into();
            match self.check_relation(polar) {
                Some(Ordering::Less) | None => return polar,
                Some(Ordering::Greater) => break,
                Some(Ordering::Equal) => {}
            }
        }
        // 选需要转向更小的一边
        let left = self.front.last().unwrap();
        let right = self.back.last().unwrap();
        if *trend * left.theta.abs() / left.rho < right.theta.abs() / right.rho {
            *trend *= 0.998;
            Polar {
                rho: left.rho,
                theta: left.theta + 0.1 / left.rho,
            }
        } else {
            *trend *= 1.002;
            Polar {
                rho: right.rho,
                theta: right.theta - 0.1 / right.rho,
            }
        }
    }

    /// 判断一个点与障碍物的关系：在前方/在内部/在后方
    fn check_relation(&self, polar: Polar) -> Option<Ordering> {
        let left = self.front.last().unwrap();
        let right = self.back.last().unwrap();
        if polar.theta < right.theta || left.theta < polar.theta {
            return None;
        }
        let ord_front = match self
            .front
            .binary_search_by(|probe| probe.theta.partial_cmp(&polar.theta).unwrap())
        {
            Ok(i) => polar.rho.partial_cmp(&self.front[i].rho).unwrap(),
            Err(i) => {
                let right = if i == 0 { *right } else { self.front[i - 1] };
                cmp(polar, (right, self.front[i]))
            }
        };
        if ord_front != Ordering::Greater {
            return Some(ord_front);
        }
        let ord_back = match self
            .back
            .binary_search_by(|probe| polar.theta.partial_cmp(&probe.theta).unwrap())
        {
            Ok(i) => polar.rho.partial_cmp(&self.back[i].rho).unwrap(),
            Err(i) => {
                let left = if i == 0 { *left } else { self.back[i - 1] };
                cmp(polar, (self.back[i], left))
            }
        };
        return Some(if ord_back != Ordering::Greater {
            Ordering::Equal
        } else {
            Ordering::Greater
        });
    }
}

/// 叉积 === 两向量围成平行四边形面积
#[inline]
fn cross_numeric(v0: Vector2<f32>, v1: Vector2<f32>) -> f32 {
    v0[1] * v1[0] - v0[0] * v1[1]
}

/// `c` 在 `ab` 左侧
#[inline]
fn is_left(a: Point2<f32>, b: Point2<f32>, c: Point2<f32>) -> bool {
    cross_numeric(b - a, c - b).is_sign_negative()
}

#[inline]
fn cmp(c: Polar, seg: (Polar, Polar)) -> Ordering {
    let a = Point2::from(seg.0);
    let b = Point2::from(seg.1);
    let c = Point2::from(c);
    cross_numeric(b - a, c - b).partial_cmp(&0.0).unwrap()
}

#[inline]
const fn isometry(x: f32, y: f32, cos: f32, sin: f32) -> Isometry2<f32> {
    use nalgebra::Translation;
    Isometry2 {
        translation: Translation {
            vector: vector(x, y),
        },
        rotation: angle(cos, sin),
    }
}

#[inline]
const fn vector(x: f32, y: f32) -> Vector2<f32> {
    use nalgebra::{ArrayStorage, OVector};
    OVector::from_array_storage(ArrayStorage([[x, y]]))
}

#[inline]
const fn angle(cos: f32, sin: f32) -> UnitComplex<f32> {
    use nalgebra::{Complex, Unit};
    Unit::new_unchecked(Complex { re: cos, im: sin })
}

#[test]
fn test_is_left() {
    const A: Point2<f32> = Point2 {
        coords: vector(0.0, 0.0),
    };
    const B: Point2<f32> = Point2 {
        coords: vector(1.0, 0.0),
    };
    const C: Point2<f32> = Point2 {
        coords: vector(2.0, 1.0),
    };
    const D: Point2<f32> = Point2 {
        coords: vector(2.0, -1.0),
    };
    assert!(is_left(A, B, C));
    assert!(!is_left(A, B, D));
}
