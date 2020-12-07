#include <jni.h>
#include <string>
#include "libopencl-stub/include/libopencl.h"
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <camera/NDK

#define LOG_TAG "OPENCL"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGW(...)  __android_log_print(ANDROID_LOG_WARN,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define MSTRINGIFY(...) #__VA_ARGS__

cl_int status;

void checkStatus(int status);
cl_program createProgramFromFile(AAssetManager *manager, cl_context context, const char *file_name);

extern "C" JNIEXPORT jstring JNICALL
Java_com_copycat_matting_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */,
        jobject assetManager) {

    setenv("LIBOPENCL_SO_PATH", "/vendor/lib64/libOpenCL.so", 1);

    cl_platform_id cpPlatform;
    cl_device_id deviceId;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_uint numPlatforms;
    cl_uint numDevices;

    AAssetManager *manager;

    status = clGetPlatformIDs(1, &cpPlatform, &numPlatforms);
    checkStatus(status);
    LOGI("Number of platforms: %d", numPlatforms);

    checkStatus(clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices));
    LOGI("Number of devices: %d", numDevices);
    checkStatus(clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_ALL, 1, &deviceId, NULL));

    context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &status);
    checkStatus(status);

    queue = clCreateCommandQueue(context, deviceId, 0, &status);
    checkStatus(status);


    manager = AAssetManager_fromJava(env, assetManager);
    program = createProgramFromFile(manager, context, "kernels/test.cl");
    checkStatus(status);

    checkStatus(clBuildProgram(program, 1, &deviceId, NULL, NULL, NULL));

    kernel = clCreateKernel(program, "testKernel", &status);
    checkStatus(status);

    return env->NewStringUTF(std::to_string(status).c_str());
}

void checkStatus(int status) {
    if (status != CL_SUCCESS) {
      LOGE("[%s:%d] OpenCL Error %d", __FILE__, __LINE__, status);
    }
}

cl_program createProgramFromFile(AAssetManager *manager, cl_context context, const char *file_name) {
    size_t sourceSize;

    AAsset *kernelAsset = AAssetManager_open(manager, file_name, AASSET_MODE_UNKNOWN);
    if (kernelAsset == NULL) {
        LOGE("Failed to open file.\n");
        exit(1);
    }

    sourceSize = AAsset_getLength(kernelAsset);
    char *sourceCode = (char *)malloc(sourceSize + 1);
    AAsset_read(kernelAsset, sourceCode, sourceSize);
    sourceCode[sourceSize] = '\0';
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&sourceCode, &sourceSize, &status);
    checkStatus(status);
    return program;
}