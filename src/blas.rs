extern crate rust_blas;

use rust_blas::Dot;

pub fn dot(v1: &Vec<f64>, v2: &Vec<f64>) -> f64 {
    Dot::dot(v1, v2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot() {
        // Given
        let x = vec![1.0, -2.0, 3.0, 4.0];
        let y = [1.0, 1.0, 1.0, 1.0, 7.0];

        // When
        let d = Dot::dot(&x, &y[..x.len()]);

        // Then
        assert_eq!(d, 6.0);
    }
}