#ifndef COV_SETTING_H
#define COV_SETTING_H

enum CovSetting {
  standard = 0,
  diag_time = 1,
  noiseless = 2,
  noiseless_profiled = 3,
};

static const bool IsNoiseless(CovSetting cov_setting) {
  return cov_setting == CovSetting::noiseless ||
         cov_setting == CovSetting::noiseless_profiled;
}

#endif