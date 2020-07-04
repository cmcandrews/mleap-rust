use crate::blas;

pub struct LinearRegressionModel {
    coefficients: Vec<f64>,
    intercept: f64,
}

impl LinearRegressionModel {
    pub fn predict(&self, features: &Vec<f64>) -> f64 {
        blas::dot(features, &self.coefficients) + self.intercept
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply() {
        // Given
        let linear_regression = LinearRegressionModel {
            coefficients: vec![0.5, 0.75, 0.25],
            intercept: 0.33,
        };
        let features = vec![1.0, 0.5, 1.0];

        // When
        let result = linear_regression.predict(&features);

        // Then
        assert_eq!(result, 1.455);
    }
}