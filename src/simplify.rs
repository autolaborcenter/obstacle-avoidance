use super::{is_left, isometry};
use crate::{Point2, Polar};
use std::f32::consts::FRAC_PI_6;

/// 同时分割并求凸包
pub fn convex_from_origin(
    polar: impl IntoIterator<Item = Polar>,
    width: f32,
) -> Vec<Vec<Point2<f32>>> {
    let mut iter = polar.into_iter().map(|p| Point2::from(p));
    let mut last = match iter.next() {
        Some(p) => p,
        None => return vec![],
    };
    let mut result = vec![vec![last]];
    let squared = width.powi(2);
    for p in iter {
        // 判断与上一个点形成的狭缝能否通过
        // 可以通过，分割
        if (p - last).norm_squared() > squared {
            result.push(vec![p]);
        }
        // 不可通过，计算凸包
        else {
            let tail = result.last_mut().unwrap();
            loop {
                match tail.as_slice() {
                    &[.., t1, t0] => {
                        if is_left(t1, t0, p) {
                            tail.pop();
                        } else {
                            break;
                        }
                    }
                    _ => break,
                }
            }
            tail.push(p);
        }
        last = p;
    }
    result
}

/// 线段拟合
pub fn fit(src: Vec<Point2<f32>>, radius: f32, max_len: f32) -> Vec<Point2<f32>> {
    match src.len() {
        0 => return src,
        1 => return vec![src[0], point_from(radius, src[0][0].atan2(src[0][1]))],
        _ => {}
    }

    let back = src[src.len() - 1];
    let back = back[1].atan2(back[0]);
    let front = src[0];
    let front = front[1].atan2(front[0]);
    let n = ((back - front) / FRAC_PI_6).round() as usize + 1;

    let mut result = Vec::with_capacity(src.len() + 2 + n);

    let step = (back - front) / n as f32;
    result.extend((0..n).map(|i| point_from(radius, back - i as f32 * step)));
    result.push(point_from(radius, front));

    let mut buf = Vec::<Point2<f32>>::new();
    for p in src {
        if !buf.is_empty() {
            let dir = (result[result.len() - 1] - p).normalize();
            let tr = isometry(p[0], p[1], dir[0], dir[1]).inverse();
            let mut min = 0.0;
            let mut max = 0.0;
            // 验证线段上所有点仍在线段上
            for p in buf.iter().rev() {
                let y = (tr * p)[1];
                if y < min {
                    min = y;
                } else if y > max {
                    max = y;
                } else {
                    continue;
                };
                // 超界，折断线段
                if max - min > max_len {
                    result.push(*buf.last().unwrap());
                    buf.clear();
                    break;
                }
            }
        }
        buf.push(p);
    }
    if let Some(p) = buf.pop() {
        result.push(p);
    }
    result
}

#[inline]
fn point_from(rho: f32, theta: f32) -> Point2<f32> {
    Polar { rho, theta }.into()
}

// Melkman 快速凸包算法，用于任意简单且封闭的多边形
// fn melkman(src: Vec<Point2<f32>>) -> Vec<Point2<f32>> {
//     if src.len() <= 3 {
//         src
//     } else {
//         let mut buf = VecDeque::from_iter(src.iter().take(3).cloned());
//         for p in src.into_iter().skip(3) {
//             // 复制最后一点形成闭环
//             let last = *buf.back().unwrap();
//             buf.push_front(last);
//             // 新点在队头线段的左侧
//             while !is_left(buf[1], buf[0], p) {
//                 buf.pop_front();
//             }
//             // 新点在队尾线段的右侧
//             while is_left(buf[buf.len() - 2], buf[buf.len() - 1], p) {
//                 buf.pop_back();
//             }
//             // 添加新点
//             buf.push_back(p);
//         }
//         buf.into_iter().collect()
//     }
// }
