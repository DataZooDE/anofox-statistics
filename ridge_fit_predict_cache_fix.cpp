// This is a patch file showing the model caching fix for ridge_fit_predict.cpp
// Replace lines 108-233 with this code:

	// Build frame signature for caching (detect if frame changes across rows)
	vector<idx_t> frame_indices;
	for (const auto &frame : subframes) {
		for (idx_t frame_idx = frame.start; frame_idx < frame.end; frame_idx++) {
			if (frame_idx < all_y.size() && !std::isnan(all_y[frame_idx]) && !all_x[frame_idx].empty()) {
				frame_indices.push_back(frame_idx);
			}
		}
	}

	// Check if model is cached and frame matches
	bool use_cache = false;
	if (state.cache.model_cache && state.cache.model_cache->initialized) {
		if (state.cache.model_cache->train_indices == frame_indices) {
			use_cache = true;
		}
	}

	// Get model parameters (either from cache or by fitting)
	double intercept;
	Eigen::VectorXd beta;
	Eigen::VectorXd x_means;
	Eigen::MatrixXd XtX_inv;
	idx_t n_train;
	idx_t df_residual;
	double mse;
	idx_t p = n_features;

	if (use_cache) {
		// Use cached model - avoids refitting for every row!
		intercept = state.cache.model_cache->intercept;
		beta = state.cache.model_cache->coefficients;
		x_means = state.cache.model_cache->x_means;
		XtX_inv = state.cache.model_cache->XtX_inv;
		n_train = state.cache.model_cache->n_train;
		df_residual = state.cache.model_cache->df_residual;
		mse = state.cache.model_cache->mse;

		if (rid < 3) {
			std::cerr << "[DEBUG Ridge] Using cached model for row " << rid << std::endl;
		}
	} else {
		// Fit new model
		n_train = frame_indices.size();

		if (rid < 3) {
			std::cerr << "[DEBUG Ridge] Fitting new model: n_train=" << n_train << ", p=" << p << std::endl;
		}

		// Validation
		idx_t min_required = p + (options.intercept ? 1 : 0) + 1;
		if (n_train < min_required || p == 0) {
			result_validity.SetInvalid(rid);
			return;
		}

		// Build training matrices from frame indices
		Eigen::MatrixXd X_train(n_train, p);
		Eigen::VectorXd y_train(n_train);

		for (idx_t i = 0; i < n_train; i++) {
			idx_t data_idx = frame_indices[i];
			y_train(i) = all_y[data_idx];
			for (idx_t col = 0; col < p; col++) {
				X_train(i, col) = all_x[data_idx][col];
			}
		}

		// Fit Ridge regression
		idx_t rank;
		if (options.intercept) {
			double y_mean = y_train.mean();
			x_means = X_train.colwise().mean();

			Eigen::VectorXd y_centered = y_train.array() - y_mean;
			Eigen::MatrixXd X_centered = X_train;
			for (idx_t j = 0; j < p; j++) {
				X_centered.col(j).array() -= x_means(j);
			}

			Eigen::MatrixXd XtX = X_centered.transpose() * X_centered;
			Eigen::VectorXd Xty = X_centered.transpose() * y_centered;
			Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(p, p);
			Eigen::MatrixXd XtX_regularized = XtX + options.lambda * identity;

			Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(XtX_regularized);
			beta = qr.solve(Xty);
			rank = qr.rank();

			intercept = y_mean - beta.dot(x_means);

			// Compute XtX_inv for prediction intervals
			Eigen::BDCSVD<Eigen::MatrixXd> svd(XtX, Eigen::ComputeThinU | Eigen::ComputeThinV);
			XtX_inv = svd.solve(Eigen::MatrixXd::Identity(p, p));
		} else {
			x_means = Eigen::VectorXd::Zero(p);

			Eigen::MatrixXd XtX = X_train.transpose() * X_train;
			Eigen::VectorXd Xty = X_train.transpose() * y_train;
			Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(p, p);
			Eigen::MatrixXd XtX_regularized = XtX + options.lambda * identity;

			Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(XtX_regularized);
			beta = qr.solve(Xty);
			rank = qr.rank();

			intercept = 0.0;

			Eigen::BDCSVD<Eigen::MatrixXd> svd(XtX, Eigen::ComputeThinU | Eigen::ComputeThinV);
			XtX_inv = svd.solve(Eigen::MatrixXd::Identity(p, p));
		}

		// Compute MSE
		Eigen::VectorXd y_pred_train = Eigen::VectorXd::Constant(n_train, intercept);
		y_pred_train += X_train * beta;
		Eigen::VectorXd residuals = y_train - y_pred_train;
		double ss_res = residuals.squaredNorm();

		idx_t df_model = rank + (options.intercept ? 1 : 0);
		df_residual = n_train - df_model;
		mse = (df_residual > 0) ? (ss_res / df_residual) : std::numeric_limits<double>::quiet_NaN();

		// Cache the model for reuse across rows
		if (!state.cache.model_cache) {
			state.cache.model_cache = new OlsModelCache();
		}
		state.cache.model_cache->coefficients = beta;
		state.cache.model_cache->intercept = intercept;
		state.cache.model_cache->mse = mse;
		state.cache.model_cache->XtX_inv = XtX_inv;
		state.cache.model_cache->x_means = x_means;
		state.cache.model_cache->df_residual = df_residual;
		state.cache.model_cache->rank = rank;
		state.cache.model_cache->n_train = n_train;
		state.cache.model_cache->train_indices = frame_indices;
		state.cache.model_cache->initialized = true;

		if (rid < 3) {
			std::cerr << "[DEBUG Ridge] Model fitted and cached: intercept=" << intercept
			          << ", beta[0]=" << (p > 0 ? beta(0) : 0.0) << std::endl;
		}
	}

	// Predict for current row
	if (rid >= all_x.size() || all_x[rid].empty()) {
		result_validity.SetInvalid(rid);
		return;
	}

	// Compute prediction with interval using cached model parameters
	PredictionResult pred = ComputePredictionWithIntervalXtXInv(all_x[rid], intercept, beta, mse, x_means, XtX_inv,
	                                                            n_train, df_residual, 0.95, "prediction");
