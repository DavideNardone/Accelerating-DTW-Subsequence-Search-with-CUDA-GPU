# Process this file with autoconf to produce a configure script.
AC_PREREQ([2.69])
AC_INIT([MTSS-Multivariate-Time-Series-Software], [1.0.0], [davide.nardone@live.it])
AC_CONFIG_SRCDIR([include/header.h])
AC_CONFIG_HEADERS([config.h])
AM_INIT_AUTOMAKE

# Checks for programs.
AC_PROG_CC
AC_PROG_CXX
AM_PROG_CC_C_O

# Check for libraries.

# Checks for header files.
AC_CHECK_HEADERS([float.h stdlib.h string.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 tests/Makefile])

# Checks for library functions.

############################################
# NVIDIA Cuda Compiler detection and setup #
############################################

# If cuda is requested to be enabled
AC_ARG_ENABLE(cuda, 

	AS_HELP_STRING([--enable-cuda=ARCH], [Enable cuda based modules for architecture ARCH (see nvcc option -arch).]),[

	# Search nvcc compiler
	AC_PATH_PROG(NVCC, nvcc, "no")

	AS_IF([test "x$NVCC" = "xno"],[

			AC_MSG_ERROR([NVCC compiler not found!])

	])


	# Check nvcc version, should be 3.0
	AS_IF([nvcc --version | grep -q "release 3.0"],

			[],

			[AC_MSG_WARN([NVCC compiler version is NOT 3.0!])
	])


	# If $with_cuda is not empty, set to CUDA_ARCH to
	# supplied value, else set to value sm_11
	AS_IF([test "x$enableval" = "xyes"],[

			CUDA_ARCH=" -arch=sm_11"

	],[

		CUDA_ARCH=" -arch=$enableval"

	])


		# Set CUDA_CFLAGS to $NVCC, where substring "bin/nvcc"
		# is substituted by "include".
		CUDA_CFLAGS=" -I${NVCC/'bin/nvcc'/include}"


		#Set CUDA_CFLAGS to $NVCC, where substring "bin/nvcc"
		#is substituted by "lib".
		CUDA_LIBS=" -L${NVCC/'bin/nvcc'/lib}"


		# If $build_cpu contains "_64", append "64" to CUDA_LIBS

	AS_IF([echo $build_cpu | grep -q "_64"],

		[CUDA_LIBS+="64"])

		# Append " -lcudart" to CUDA_LIBS

		CUDA_LIBS+=" -lcudart"


	# Symbolize that cuda is wanted
	with_cuda=$enableval

	# Make variables available in Makefile.am
	AC_SUBST(CUDA_CFLAGS)
	AC_SUBST(CUDA_LIBS)
	AC_SUBST(NVCC)
])

# Set this conditional if cuda is wanted
AM_CONDITIONAL([WANT_CUDA], [test -n "$with_cuda"])


# Check whether to use device emulation mode for cuda (if no cuda capable gpu is available)
AC_ARG_ENABLE([emu],

	AS_HELP_STRING([--enable-emu], [Enable device emulation for cuda modules (nvcc version <= 3.0 only).]),

	[EMULATION=true],

	[EMULATION=false])


# Set up compilation flags for cuda compiler nvcc, if with_cuda is set
AS_IF([test -n "$with_cuda"],[

	# If debug flag is set apply debugging compilation flags, otherwise build compilation flags
	AS_IF([test "x$DEBUG" = "xtrue"],

		[NVCCFLAGS="-g --compiler-options -fno-strict-aliasing --compiler-options -fno-inline"],

		[NVCCFLAGS="-O3 -use_fast_math --compiler-options -fno-strict-aliasing --compiler-options -fno-inline"])

	# Add architecture to flags
	NVCCFLAGS+=" $CUDA_ARCH"

	# If device emulation was set, add deviceemu flag
	AS_IF([test "x$EMULATION" = "xtrue"],

		[NVCCFLAGS+=" -deviceemu"])

])

# Make NVCCFLAGS available in Makefile.am
AC_SUBST(NVCCFLAGS)

############################################
# NVIDIA Cuda Compiler detection and setup #
############################################

AC_OUTPUT