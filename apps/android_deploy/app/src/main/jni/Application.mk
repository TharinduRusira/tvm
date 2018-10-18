ifndef config
	ifneq ("$(wildcard ./config.mk)","")
	  config ?= config.mk
	else
	  config ?= make/config.mk
	endif
endif

include $(config)

<<<<<<< HEAD
APP_STL := gnustl_static
=======
APP_STL := c++_static
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

APP_CPPFLAGS += -DDMLC_LOG_STACK_TRACE=0 -DTVM4J_ANDROID=1 -std=c++11 -Oz -frtti
ifeq ($(USE_OPENCL), 1)                                                                                                                                             
	APP_CPPFLAGS += -DTVM_OPENCL_RUNTIME=1
endif
