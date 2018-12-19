# Process this file with autoconf to produce a configure script.
AC_PREREQ([2.69])
AC_INIT([MTSS-Multivariate-Time-Series-Software], [1.0.0], [davide.nardone@live.it])
AC_CONFIG_SRCDIR([include/header.h])
AC_CONFIG_HEADERS([config.h])
AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects])

# Checks for programs.
AC_PROG_CC

# Checks for libraries.

# Checks for header files.
AC_CHECK_HEADERS([float.h stdlib.h string.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T

AC_CONFIG_FILES([Makefile])

# Checks for library functions.
AC_OUTPUT