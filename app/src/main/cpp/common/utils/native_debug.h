#include <android/log.h>

#define LOG_TAG "MATTING"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define ASSERT(cond, fmt, ...)                                \
  if (!(cond)) {                                              \
    __android_log_assert(#cond, LOG_TAG, fmt, ##__VA_ARGS__); \
  }

#define checkStatus(status) {                                      \
    if (status != CL_SUCCESS) {                                    \
      LOGE("[%s:%d] OpenCL Error %d", __FILE__, __LINE__, status); \
    }                                                              \
}