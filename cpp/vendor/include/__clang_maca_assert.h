/*
 * Minimal MACA/cu-bridge compatibility shim.
 *
 * Some MACA CCCL headers include this clang-provided header in Release builds.
 * Host-side assert declarations already come from <assert.h>; keeping this shim
 * local avoids adding MACA clang resource headers to every host compilation.
 */
#pragma once

#include <assert.h>
