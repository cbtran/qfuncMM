#ifndef COV_SETTING_H
#define COV_SETTING_H

enum CovSetting { noisy = 0, noiseless = 2 };

static const bool IsNoiseless(CovSetting cov_setting) {
  return cov_setting == CovSetting::noiseless;
}

#endif