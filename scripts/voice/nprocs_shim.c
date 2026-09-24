// nprocs_shim — LD_PRELOAD interposer that makes get_nprocs()/get_nprocs_conf()
// (and therefore std::thread::hardware_concurrency()) return $SHIM_NPROCS.
//
// WHY: qwentts.cpp @ production-speech-v1 has no thread-count flag; src/backend.h
// hardcodes hardware_concurrency()/2, which ignores CPU affinity, so a CPU TTS
// pinned to 16 cores would still start 96 spinning threads on this host. The
// speech tree is FROZEN, so the count is set from OUTSIDE it: SHIM_NPROCS=2N gives
// N threads. Unset or <= 0 -> the real libc answer (the shim is then inert).
//
// This is the exact source that produced the 2026-09-24 CPU speech measurements
// (epyc-inference-research artifacts/speech_cpu_realtime_20260924/nprocs_shim.c);
// do not change its behaviour without re-measuring. The durable fix is a thread
// flag in qwentts.cpp on an experimental branch -> the next speech kernel version.
// Build: scripts/voice/build_nprocs_shim.sh (out of every frozen tree).
#define _GNU_SOURCE
#include <stdlib.h>
#include <dlfcn.h>
static int val(void){ const char*s=getenv("SHIM_NPROCS"); return s?atoi(s):-1; }
int get_nprocs(void){ int v=val(); if(v>0) return v; int(*f)(void)=dlsym(RTLD_NEXT,"get_nprocs"); return f();}
int get_nprocs_conf(void){ int v=val(); if(v>0) return v; int(*f)(void)=dlsym(RTLD_NEXT,"get_nprocs_conf"); return f();}
