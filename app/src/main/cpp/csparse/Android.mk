LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)
LOCAL_MODULE := csparse
LOCAL_C_INCLUDES := $(LOCAL_PATH)/include/
LOCAL_SRC_FILES := src/csparse.c

include $(BUILD_STATIC_LIBRARY)
