#ifndef COV_SETTING_H
#define COV_SETTING_H

#include <string>
#include <unordered_map>

enum CovSetting {
  standard = 0,
  diag_time = 1,
  noiseless = 2,
  noiseless_profiled = 3,
};

static const std::unordered_map<std::string, CovSetting> strToCovSettingMap = {
    {"standard", standard},
    {"diag_time", diag_time},
    {"noiseless", noiseless},
    {"noiseless_profiled", noiseless_profiled}};

#endif