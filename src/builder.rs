use crate::Polar;
use std::{cmp::Ordering, f32::consts::PI};

pub(super) struct ObstacleBuilder {
    buf: [Vec<Polar>; 3],
    dir: Ordering,
    step: usize,
    last: f32,
    offset: f32,
}

impl ObstacleBuilder {
    #[inline]
    pub fn new(head: f32) -> Self {
        Self {
            buf: [vec![], vec![], vec![]],
            dir: Ordering::Equal,
            step: 0,
            last: head,
            offset: 0.0,
        }
    }

    pub fn push(&mut self, mut p: Polar) {
        let diff = p.theta + self.offset - self.last;
        if diff > PI {
            self.offset -= 2.0 * PI;
        } else if diff < -PI {
            self.offset += 2.0 * PI;
        }
        p.theta += self.offset;

        let next = p.theta.partial_cmp(&self.last).unwrap();
        if self.dir != Ordering::Equal && self.dir != next {
            self.step += 1;
        }
        self.dir = next;

        self.buf[self.step].push(p);
        self.last = p.theta;
    }

    #[inline]
    pub fn collect(self) -> (Vec<Polar>, Vec<Polar>) {
        let [a, b, mut c] = self.buf;
        c.extend(a);
        (b, c)
    }
}
