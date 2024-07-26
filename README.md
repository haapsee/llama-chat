# llama-chat

## Install with 

### CUDA

Add into .env file `CMAKE_ARGS="-DGGML_CUDA=on"`

### Vulkan

Add into .env file `CMAKE_ARGS="-DGGML_VULKAN=on"`

### OpenBLAS

Add into .env file `CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"`
