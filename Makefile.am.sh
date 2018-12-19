AUTOMAKE_OPTIONS = foreign
SUBDIRS = src/
EXTRA_DIST = autogen.sh

bin_PROGRAMS = mdtwObj

mdtwObj_SOURCES = module.cu MD_DTW.cu ../include/header.h