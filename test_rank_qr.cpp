#include <Eigen/Dense>
#include <iostream>
#include <cmath>

int main() {
    // Test case: x1=[1,2,3,4,5], x2=2*x1, y=2*x1+1 = [3,5,7,9,11]
    // After centering: y_c = y - 7 = [-4,-2,0,2,4]
    //                  x1_c = x1 - 3 = [-2,-1,0,1,2]
    //                  x2_c = x2 - 6 = [-4,-2,0,2,4]
    // x2_c = 2 * x1_c (perfect collinearity)

    Eigen::MatrixXd X(5, 2);
    Eigen::VectorXd y(5);

    // Centered data
    X << -2, -4,
         -1, -2,
          0,  0,
          1,  2,
          2,  4;

    y << -4, -2, 0, 2, 4;

    std::cout << "X (centered):\n" << X << "\n\n";
    std::cout << "y (centered):\n" << y << "\n\n";

    // QR decomposition with column pivoting
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X);

    std::cout << "Rank: " << qr.rank() << " (expected: 1)\n";
    std::cout << "Threshold: " << qr.threshold() << "\n";
    std::cout << "Permutation: " << qr.colsPermutation().indices().transpose() << "\n\n";

    // Extract R matrix to see singular values
    Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
    std::cout << "R (upper triangular from QR):\n" << R << "\n\n";
    std::cout << "R diagonal: " << R.diagonal().transpose() << "\n\n";

    // Solve using only rank-1 subsystem
    if (qr.rank() > 0) {
        Eigen::VectorXd QtY = qr.matrixQ().transpose() * y;
        std::cout << "Q^T * y: " << QtY.transpose() << "\n\n";

        Eigen::MatrixXd R_reduced = qr.matrixQR().topLeftCorner(qr.rank(), qr.rank());
        Eigen::VectorXd coef_reduced = R_reduced.triangularView<Eigen::Upper>().solve(QtY.head(qr.rank()));

        std::cout << "coef_reduced (rank=" << qr.rank() << "): " << coef_reduced.transpose() << "\n\n";

        // Map back to original columns
        Eigen::VectorXd coef = Eigen::VectorXd::Constant(2, std::nan(""));
        for (int i = 0; i < qr.rank(); i++) {
            int original_idx = qr.colsPermutation().indices()[i];
            coef[original_idx] = coef_reduced[i];
        }

        std::cout << "Coefficients (with NaN for aliased):\n";
        std::cout << "  coef[0] (x1): " << coef[0] << "\n";
        std::cout << "  coef[1] (x2): " << coef[1] << "\n\n";

        // Compute predictions
        Eigen::VectorXd y_pred = Eigen::VectorXd::Zero(5);
        for (int j = 0; j < 2; j++) {
            if (!std::isnan(coef[j])) {
                y_pred += coef[j] * X.col(j);
            }
        }

        std::cout << "y_pred: " << y_pred.transpose() << "\n";
        std::cout << "y:      " << y.transpose() << "\n";
        std::cout << "residuals: " << (y - y_pred).transpose() << "\n";
        std::cout << "SS_res: " << (y - y_pred).squaredNorm() << "\n\n";

        // Compute intercept using means
        double mean_y = 7.0;  // mean of [3,5,7,9,11]
        Eigen::VectorXd x_means(2);
        x_means << 3.0, 6.0;  // means of x1 and x2

        double beta_dot_xmean = 0.0;
        for (int j = 0; j < 2; j++) {
            if (!std::isnan(coef[j])) {
                beta_dot_xmean += coef[j] * x_means[j];
            }
        }
        double intercept = mean_y - beta_dot_xmean;

        std::cout << "Intercept calculation:\n";
        std::cout << "  mean_y: " << mean_y << "\n";
        std::cout << "  x_means: " << x_means.transpose() << "\n";
        std::cout << "  beta_dot_xmean: " << beta_dot_xmean << "\n";
        std::cout << "  intercept: " << intercept << " (expected: 1.0)\n";
    }

    return 0;
}
