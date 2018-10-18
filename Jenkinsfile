#!groovy
// -*- mode: groovy -*-
// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// tvm libraries
tvm_runtime = "build/libtvm_runtime.so, build/config.cmake"
tvm_lib = "build/libtvm.so, " + tvm_runtime
// LLVM upstream lib
<<<<<<< HEAD
tvm_multilib = "lib/libtvm_llvm40.so, lib/libtvm_llvm50.so, lib/libtvm_llvm60.so, " +
             "lib/libtvm_topi.so, nnvm/lib/libnnvm_compiler.so, " + tvm_runtime
=======
tvm_multilib = "build/libtvm.so, " +
             "build/libvta.so, build/libtvm_topi.so, build/libnnvm_compiler.so, " + tvm_runtime
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

// command to start a docker container
docker_run = 'docker/bash.sh'
// timeout in minutes
max_time = 60

// initialize source codes
def init_git() {
  checkout scm
  retry(5) {
    timeout(time: 2, unit: 'MINUTES') {
      sh 'git submodule update --init'
    }
  }
}

def init_git_win() {
    checkout scm
    retry(5) {
        timeout(time: 2, unit: 'MINUTES') {
            bat 'git submodule update --init'
        }
    }
}

stage("Sanity Check") {
  timeout(time: max_time, unit: 'MINUTES') {
    node('CPU') {
      ws('workspace/tvm/sanity') {
        init_git()
        sh "${docker_run} tvmai/ci-lint  ./tests/scripts/task_lint.sh"
      }
    }
  }
}

// Run make. First try to do an incremental make from a previous workspace in hope to
// accelerate the compilation. If something wrong, clean the workspace and then
// build from scratch.
def make(docker_type, path, make_flag) {
  timeout(time: max_time, unit: 'MINUTES') {
    try {
<<<<<<< HEAD
      sh "${docker_run} ${docker_type} ./tests/scripts/task_build.sh ${make_flag}"
    } catch (exc) {
      echo 'Incremental compilation failed. Fall back to build from scratch'
      sh "${docker_run} ${docker_type} ./tests/scripts/task_clean.sh"
      sh "${docker_run} ${docker_type} ./tests/scripts/task_build.sh ${make_flag}"
=======
      sh "${docker_run} ${docker_type} ./tests/scripts/task_build.sh ${path} ${make_flag}"
    } catch (exc) {
      echo 'Incremental compilation failed. Fall back to build from scratch'
      sh "${docker_run} ${docker_type} ./tests/scripts/task_clean.sh ${path}"
      sh "${docker_run} ${docker_type} ./tests/scripts/task_build.sh ${path} ${make_flag}"
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
    }
  }
}

// pack libraries for later use
def pack_lib(name, libs) {
  sh """
     echo "Packing ${libs} into ${name}"
     echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
     """
  stash includes: libs, name: name
}


// unpack libraries saved before
def unpack_lib(name, libs) {
  unstash name
  sh """
     echo "Unpacked ${libs} from ${name}"
     echo ${libs} | sed -e 's/,/ /g' | xargs md5sum
     """
}

stage('Build') {
  parallel 'BUILD: GPU': {
    node('GPUBUILD') {
      ws('workspace/tvm/build-gpu') {
        init_git()
        sh """
<<<<<<< HEAD
           cp make/config.mk .
           echo USE_CUDNN=1 >> config.mk
           echo USE_CUDA=1 >> config.mk
           echo USE_OPENGL=1 >> config.mk
           echo LLVM_CONFIG=llvm-config-4.0 >> config.mk
           echo USE_RPC=1 >> config.mk
           echo USE_SORT=1 >> config.mk
           echo USE_GRAPH_RUNTIME=1 >> config.mk
           echo USE_BLAS=openblas >> config.mk
           rm -f lib/libtvm_runtime.so lib/libtvm.so
=======
           mkdir -p build
           cd build
           cp ../cmake/config.cmake .
           echo set\\(USE_CUBLAS ON\\) >> config.cmake
           echo set\\(USE_CUDNN ON\\) >> config.cmake
           echo set\\(USE_CUDA ON\\) >> config.cmake
           echo set\\(USE_OPENGL ON\\) >> config.cmake
           echo set\\(USE_LLVM llvm-config-6.0\\) >> config.cmake
           echo set\\(USE_RPC ON\\) >> config.cmake
           echo set\\(USE_SORT ON\\) >> config.cmake
           echo set\\(USE_GRAPH_RUNTIME ON\\) >> config.cmake
           echo set\\(USE_STACKVM_RUNTIME ON\\) >> config.cmake
           echo set\\(USE_GRAPH_RUNTIME_DEBUG ON\\) >> config.cmake
           echo set\\(USE_BLAS openblas\\) >> config.cmake
           echo set\\(CMAKE_CXX_COMPILER g++\\) >> config.cmake
           echo set\\(CMAKE_CXX_FLAGS -Werror\\) >> config.cmake
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
           """
        make('tvmai/ci-gpu', 'build', '-j2')
        pack_lib('gpu', tvm_multilib)
        // compiler test
        sh """
<<<<<<< HEAD
           echo USE_OPENCL=1 >> config.mk
           echo USE_ROCM=1 >> config.mk
           echo ROCM_PATH=/opt/rocm >> config.mk
           echo USE_VULKAN=1 >> config.mk
=======
           mkdir -p build2
           cd build2
           cp ../cmake/config.cmake .
           echo set\\(USE_OPENCL ON\\) >> config.cmake
           echo set\\(USE_ROCM ON\\) >> config.cmake
           echo set\\(USE_VULKAN ON\\) >> config.cmake
           echo set\\(USE_GRAPH_RUNTIME_DEBUG ON\\) >> config.cmake
           echo set\\(CMAKE_CXX_COMPILER clang-6.0\\) >> config.cmake
           echo set\\(CMAKE_CXX_FLAGS -Werror\\) >> config.cmake
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
           """
        make('tvmai/ci-gpu', 'build2', '-j2')
      }
    }
  },
  'BUILD: CPU': {
    node('CPU') {
      ws('workspace/tvm/build-cpu') {
        init_git()
        sh """
<<<<<<< HEAD
           cp make/config.mk .
           echo USE_CUDA=0 >> config.mk
           echo USE_OPENCL=0 >> config.mk
           echo USE_RPC=0 >> config.mk
           echo USE_SORT=1 >> config.mk
           echo USE_OPENGL=1 >> config.mk
           echo LLVM_CONFIG=llvm-config-4.0 >> config.mk
           echo USE_SORT=1 >> config.mk
=======
           mkdir -p build
           cd build
           cp ../cmake/config.cmake .
           echo set\\(USE_SORT ON\\) >> config.cmake
           echo set\\(USE_GRAPH_RUNTIME_DEBUG ON\\) >> config.cmake
           echo set\\(USE_LLVM llvm-config-4.0\\) >> config.cmake
           echo set\\(CMAKE_CXX_COMPILER g++\\) >> config.cmake
           echo set\\(CMAKE_CXX_FLAGS -Werror\\) >> config.cmake
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
           """
        make('tvmai/ci-cpu', 'build', '-j2')
        pack_lib('cpu', tvm_lib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} tvmai/ci-cpu ./tests/scripts/task_cpp_unittest.sh"
          sh "${docker_run} tvmai/ci-cpu ./tests/scripts/task_python_vta.sh"
          sh "${docker_run} tvmai/ci-cpu ./tests/scripts/task_rust.sh"
        }
      }
    }
  },
  'BUILD : i386': {
    node('CPU') {
      ws('workspace/tvm/build-i386') {
        init_git()
        sh """
<<<<<<< HEAD
           cp make/config.mk .
           echo USE_CUDA=0 >> config.mk
           echo USE_OPENCL=0 >> config.mk
           echo LLVM_CONFIG=llvm-config-4.0 >> config.mk
           echo USE_RPC=1 >> config.mk
           echo USE_SORT=1 >> config.mk
=======
           mkdir -p build
           cd build
           cp ../cmake/config.cmake .
           echo set\\(USE_SORT ON\\) >> config.cmake
           echo set\\(USE_RPC ON\\) >> config.cmake
           echo set\\(USE_GRAPH_RUNTIME_DEBUG ON\\) >> config.cmake
           echo set\\(USE_LLVM llvm-config-5.0\\) >> config.cmake
           echo set\\(CMAKE_CXX_COMPILER g++\\) >> config.cmake
           echo set\\(CMAKE_CXX_FLAGS -Werror\\) >> config.cmake
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
           """
        make('tvmai/ci-i386', 'build', '-j2')
        pack_lib('i386', tvm_multilib)
      }
    }
  }
}

stage('Unit Test') {
  parallel 'python2/3: GPU': {
    node('GPU') {
      ws('workspace/tvm/ut-python-gpu') {
        init_git()
        unpack_lib('gpu', tvm_multilib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} tvmai/ci-gpu ./tests/scripts/task_python_unittest.sh"
          sh "${docker_run} tvmai/ci-gpu ./tests/scripts/task_python_integration.sh"
        }
      }
    }
  },
  'python2/3: i386': {
    node('CPU') {
      ws('workspace/tvm/ut-python-i386') {
        init_git()
        unpack_lib('i386', tvm_multilib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} tvmai/ci-i386 ./tests/scripts/task_python_unittest.sh"
          sh "${docker_run} tvmai/ci-i386 ./tests/scripts/task_python_integration.sh"
          sh "${docker_run} tvmai/ci-i386 ./tests/scripts/task_python_vta.sh"
        }
      }
    }
  },
<<<<<<< HEAD
  'cpp': {
    node('CPU' && 'linux') {
      ws('workspace/tvm/ut-cpp') {
        init_git()
        unpack_lib('cpu', tvm_lib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} cpu ./tests/scripts/task_cpp_unittest.sh"
        }
      }
    }
  },
  'java': {
    node('GPU' && 'linux') {
=======
  'java: GPU': {
    node('GPU') {
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
      ws('workspace/tvm/ut-java') {
        init_git()
        unpack_lib('gpu', tvm_multilib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} tvmai/ci-gpu ./tests/scripts/task_java_unittest.sh"
        }
      }
    }
  }
}

stage('Integration Test') {
  parallel 'topi: GPU': {
    node('GPU') {
      ws('workspace/tvm/it-python-gpu') {
        init_git()
        unpack_lib('gpu', tvm_multilib)
        timeout(time: max_time, unit: 'MINUTES') {
<<<<<<< HEAD
          sh "${docker_run} gpu ./tests/scripts/task_python_integration.sh"
          sh "${docker_run} gpu ./tests/scripts/task_python_topi.sh"
          sh "${docker_run} gpu ./tests/scripts/task_cpp_topi.sh"
          sh "${docker_run} gpu ./tests/scripts/task_python_nnvm.sh"
=======
          sh "${docker_run} tvmai/ci-gpu ./tests/scripts/task_python_topi.sh"
          sh "${docker_run} tvmai/ci-gpu ./tests/scripts/task_cpp_topi.sh"
        }
      }
    }
  },
  'nnvm: GPU': {
    node('GPU') {
      ws('workspace/tvm/it-python-gpu') {
        init_git()
        unpack_lib('gpu', tvm_multilib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} tvmai/ci-gpu ./tests/scripts/task_python_nnvm.sh"
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
        }
      }
    }
  },
  'docs: GPU': {
    node('GPU') {
      ws('workspace/tvm/docs-python-gpu') {
        init_git()
        unpack_lib('gpu', tvm_multilib)
        timeout(time: max_time, unit: 'MINUTES') {
          sh "${docker_run} tvmai/ci-gpu ./tests/scripts/task_python_docs.sh"
        }
        pack_lib('mydocs', 'docs.tgz')
      }
    }
  }
}

stage('Deploy') {
    node('doc') {
      ws('workspace/tvm/deploy-docs') {
        if (env.BRANCH_NAME == "master") {
           unpack_lib('mydocs', 'docs.tgz')
           sh "tar xf docs.tgz -C /var/docs"
        }
      }
    }
}
