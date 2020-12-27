#include <jni.h>
#include <string>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <media/NdkImage.h>
#include <media/NdkImageReader.h>
#include <android/log.h>
#include "common/utils/native_debug.h"
#include "libopencl-stub/include/libopencl.h"
#include "csparse/include/csparse.h"

#define WIN_SIZE 9
#define WIN_RAD 1
#define WIN_DIAM 3

cl_int status;
cl_platform_id cpPlatform;
cl_device_id deviceId;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel laplacianKernel;
cl_uint numPlatforms;
cl_uint numDevices;

cl_program createProgramFromFile(AAssetManager *manager, cl_context context, const char *file_name);
void* loadPixelsFromImage(AAssetManager *manager, const char* file_name);

extern "C" JNIEXPORT void JNICALL
Java_com_copycat_matting_MainActivity_init(
        JNIEnv *env,
        jobject  /* this */,
        jobject assetManager) {
    setenv("LIBOPENCL_SO_PATH", "/system/vendor/lib64/libOpenCL.so", 1);
    AAssetManager *manager;
    // Get Platform
    status = clGetPlatformIDs(1, &cpPlatform, &numPlatforms);
    checkStatus(status);
    LOGI("Number of platforms: %d", numPlatforms);
    LOGI("Max size of work group: %d", CL_KERNEL_WORK_GROUP_SIZE);
    // Get devices available.
    checkStatus(clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices));
    LOGI("Number of GPU devices: %d", numDevices);
    checkStatus(clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &deviceId, NULL));
    // Create a context
    context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &status);
    checkStatus(status);
    // Create a command queue
    queue = clCreateCommandQueue(context, deviceId, 0, &status);
    checkStatus(status);
    // Retrieve kernel codes from the android asset.
    manager = AAssetManager_fromJava(env, assetManager);
    program = createProgramFromFile(manager, context, "kernels/matting.cl");
    checkStatus(status);
    // Build
    checkStatus(clBuildProgram(program, 1, &deviceId, NULL, NULL, NULL));
    // Create kernels
    laplacianKernel = clCreateKernel(program, "calculateLaplacian", &status);
    checkStatus(status);
}

extern "C" JNIEXPORT void JNICALL
Java_com_copycat_matting_MainActivity_drawTest(
        JNIEnv *env,
        jobject /* this */,
        jintArray pixels,
        jintArray trimapPixels,
        jint height,
        jint width) {
    int *hostColIndex;
    int *hostRowIndex;
    float *hostLaplacian;

    int *hostImage = env->GetIntArrayElements(pixels, NULL);
    int *hostTrimapImage = env->GetIntArrayElements(trimapPixels, NULL);

//    hostImage[0] = 0;
//    hostImage[1] = 0;
//    hostImage[2] = 0;
//    hostImage[3] = 0;
//    hostImage[4] = 0xffff00ff;
//    hostImage[5] = 0;
//    hostImage[6] = 0;
//    hostImage[7] = 0;
//    hostImage[8] = 0;
//
//    hostTrimapImage[0] = 0;
//    hostTrimapImage[1] = 0;
//    hostTrimapImage[2] = 0;
//    hostTrimapImage[3] = 0;
//    hostTrimapImage[4] = 0xffffffff;
//    hostTrimapImage[5] = 0;
//    hostTrimapImage[6] = 0;
//    hostTrimapImage[7] = 0;
//    hostTrimapImage[8] = 0;
//
//    height = 3;
//    width = 3;

    LOGI("Image received: %d, %d", height, width);
    LOGI("The total number of pixels is %d: ", env->GetArrayLength(pixels));

    int c_h = height - 2 * WIN_RAD;
    int c_w = width - 2 * WIN_RAD;

    size_t indexSpaceSize[2] = {0};
    size_t workGroupSize[2] = {16, 16};
    indexSpaceSize[0] = height;
    indexSpaceSize[1] = width;
    for (int i = 0; i < 2; i++) {
        indexSpaceSize[i] = (indexSpaceSize[i] + workGroupSize[i] - 1) / workGroupSize[i] * workGroupSize[i];
    }

    cl_mem deviceImage;
    cl_mem laplacian;
    cl_mem col_index;
    cl_mem row_index;

    LOGI("Create deviceImage buffer");
    deviceImage = clCreateBuffer(context,
                                 CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                 height * width * sizeof(int),
                                 NULL,
                                 &status);
    checkStatus(status);

    LOGI("Create laplacian buffer");
    laplacian = clCreateBuffer(context,
                               CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                               c_h * c_w * WIN_SIZE * WIN_SIZE * sizeof(float),
                               NULL,
                               &status);
    checkStatus(status);

    LOGI("Create col_index buffer");
    col_index = clCreateBuffer(context,
                               CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                               c_h * c_w * WIN_SIZE * WIN_SIZE * sizeof(int),
                               NULL,
                               &status);
    checkStatus(status);

    LOGI("Create row_index buffer");
    row_index = clCreateBuffer(context,
                               CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                               c_h * c_w * WIN_SIZE * WIN_SIZE * sizeof(int),
                               NULL,
                               &status);
    checkStatus(status);

    LOGI("write buffer");
    checkStatus(clEnqueueWriteBuffer(queue, deviceImage, CL_TRUE, 0, height * width * sizeof(int), hostImage, 0, NULL, NULL));

    LOGI("kernel arg");
    checkStatus(clSetKernelArg(laplacianKernel, 0, sizeof(cl_mem), &deviceImage));
    checkStatus(clSetKernelArg(laplacianKernel, 1, sizeof(cl_mem), &laplacian));
    checkStatus(clSetKernelArg(laplacianKernel, 2, sizeof(cl_mem), &col_index));
    checkStatus(clSetKernelArg(laplacianKernel, 3, sizeof(cl_mem), &row_index));
    checkStatus(clSetKernelArg(laplacianKernel, 4, sizeof(int), &height));
    checkStatus(clSetKernelArg(laplacianKernel, 5, sizeof(int), &width));

    cl_event kernel_event;
    LOGI("ND");
    checkStatus(clEnqueueNDRangeKernel(queue, laplacianKernel, 2, NULL, indexSpaceSize, workGroupSize, 0, NULL, &kernel_event));
    checkStatus(status);
    checkStatus(clFinish(queue));
    LOGI("finish");
    checkStatus(clWaitForEvents(1, &kernel_event));
    hostLaplacian = (float *) clEnqueueMapBuffer(queue, laplacian, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, c_h * c_w * WIN_SIZE * WIN_SIZE, 0, NULL, NULL, &status);
    checkStatus(status);
    hostColIndex = (int *) clEnqueueMapBuffer(queue, col_index, CL_TRUE, CL_MAP_READ, 0, c_h * c_w * WIN_SIZE * WIN_SIZE, 0, NULL, NULL, &status);
    checkStatus(status);
    hostRowIndex = (int *) clEnqueueMapBuffer(queue, row_index, CL_TRUE, CL_MAP_READ, 0, c_h * c_w * WIN_SIZE * WIN_SIZE, 0, NULL, NULL, &status);
    checkStatus(status);

    float *alpha;
    alpha = (float *) malloc(height * width * sizeof(float));

    for (int i = 0; i < height * width; i++) {
        if ((hostTrimapImage[i] & 0x00ffffff) > 9999999) {
            hostLaplacian[i] += 100.0f;
            alpha[i] = 100.0f;
        } else if ((hostTrimapImage[i] & 0x00ffffff) < 100) {
            hostLaplacian[i] += 100.0f;
        }
    }

    cs *L = cs_spalloc(height * width, height * width, c_h * c_w * WIN_SIZE * WIN_SIZE, 1, 1);

    for (int i = 0; i < c_h * c_w * WIN_SIZE * WIN_SIZE; i++) {
        cs_entry(L, hostRowIndex[i], hostColIndex[i], hostLaplacian[i]);
    }

    // solve linear system
    cs_lsolve(L, alpha);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int ij = i*width+j;
            int color = 0;
            if (alpha[ij] > 1) {
                hostTrimapImage[ij] = 0xffffffff;
            } else if (alpha[ij] < 0) {
                hostTrimapImage[ij] = 0xff000000;
            } else {
                color = alpha[ij] * 255;
                hostTrimapImage[ij] = 0xff000000 + color * 0xffff + color * 0xff + color;
            }
        }
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

