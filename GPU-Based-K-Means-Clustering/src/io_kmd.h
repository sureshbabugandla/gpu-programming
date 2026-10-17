// io_kmd.h — binary dataset and output files (requirements.md BLD-004, FR-002, FR-015).
//
// .kmd: magic "KMD1", uint32 version (= 1), uint64 rows, uint64 cols, then rows*cols binary32
//       little-endian in row-major order. Used for data and for centroids.
// .kma: magic "KMA1", uint32 version (= 1), uint64 N, then N int32 assignments.
//
// Every stream is opened with std::ios::binary. On Windows the default text mode translates byte
// 0x0A on write and 0x0D 0x0A on read, which silently corrupts float payloads containing those
// bytes, and the exact-size check below would then fail on a file that was written correctly.

#ifndef KMEANS_IO_KMD_H
#define KMEANS_IO_KMD_H

#include <cstdint>
#include <string>
#include <vector>

#include "common.h"

namespace kmeans {

struct Matrix {
    uint64_t rows = 0;
    uint64_t cols = 0;
    std::vector<float> data;  // row-major, rows*cols values
};

// BLD-004 requires little-endian on disk; the host must therefore be little-endian.
inline bool host_is_little_endian()
{
    const uint32_t one = 1;
    unsigned char bytes[sizeof(one)];
    std::memcpy(bytes, &one, sizeof(one));
    return bytes[0] == 1;
}

// Returns kExitOk, or kExitIoError for a missing, unreadable, truncated, over-long or
// structurally invalid file (bad magic or version). `err` receives a BLD-006 style message.
int read_kmd(const std::string &path, Matrix *out, std::string *err);
int read_kma(const std::string &path, std::vector<int32_t> *out, std::string *err);

// As read_kmd, plus the FR-002 check that the header dimensions equal the declared ones. A
// dimension mismatch is a validation failure (kExitInvalidInput), not an I/O failure.
int read_kmd_expect(const std::string &path, uint64_t rows, uint64_t cols, Matrix *out,
                    std::string *err);

int write_kmd(const std::string &path, uint64_t rows, uint64_t cols, const float *data,
              std::string *err);
int write_kma(const std::string &path, const std::vector<int32_t> &assignments, std::string *err);

}  // namespace kmeans

#endif  // KMEANS_IO_KMD_H
