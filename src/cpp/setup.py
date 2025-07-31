from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='mcpg_kernel',
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            'mcpg_kernel', # operator name
            ['./mcpg_kernel_wrapper.cpp',
             './mcpg_kernel.cu',]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)