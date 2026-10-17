#include "io_kmd.h"

#include <fstream>

namespace kmeans {
namespace {

constexpr uint32_t kFormatVersion = 1;
constexpr size_t kKmdHeaderBytes = 4 + 4 + 8 + 8;  // magic, version, rows, cols
constexpr size_t kKmaHeaderBytes = 4 + 4 + 8;      // magic, version, N

int io_fail(std::string *err, const std::string &message)
{
    if (err != nullptr) {
        *err = message;
    }
    return kExitIoError;
}

// Opens the file, returns its exact size and leaves the stream positioned at the start.
bool open_sized(const std::string &path, std::ifstream *in, uint64_t *size)
{
    in->open(path, std::ios::binary | std::ios::ate);
    if (!in->is_open()) {
        return false;
    }
    const std::streampos end = in->tellg();
    if (end < 0) {
        return false;
    }
    *size = static_cast<uint64_t>(end);
    in->seekg(0, std::ios::beg);
    return true;
}

bool read_exact(std::ifstream *in, void *dst, size_t bytes)
{
    in->read(static_cast<char *>(dst), static_cast<std::streamsize>(bytes));
    return static_cast<size_t>(in->gcount()) == bytes && !in->bad();
}

}  // namespace

int read_kmd(const std::string &path, Matrix *out, std::string *err)
{
    if (!host_is_little_endian()) {
        return io_fail(err, "host is big-endian; BLD-004 fixes little-endian byte order on disk");
    }

    std::ifstream in;
    uint64_t file_size = 0;
    if (!open_sized(path, &in, &file_size)) {
        return io_fail(err, "cannot open '" + path + "' for reading");
    }
    if (file_size < kKmdHeaderBytes) {
        return io_fail(err, "'" + path + "' is " + std::to_string(file_size) +
                                " B, shorter than the " + std::to_string(kKmdHeaderBytes) +
                                " B .kmd header");
    }

    char magic[4] = {0, 0, 0, 0};
    uint32_t version = 0;
    uint64_t rows = 0;
    uint64_t cols = 0;
    if (!read_exact(&in, magic, 4) || !read_exact(&in, &version, 4) ||
        !read_exact(&in, &rows, 8) || !read_exact(&in, &cols, 8)) {
        return io_fail(err, "'" + path + "': header could not be read");
    }
    if (std::memcmp(magic, "KMD1", 4) != 0) {
        return io_fail(err, "'" + path + "': bad magic, expected KMD1");
    }
    if (version != kFormatVersion) {
        return io_fail(err, "'" + path + "': unsupported .kmd version " + std::to_string(version));
    }

    // FR-002: the file size must equal header + payload exactly, so a truncated file and a file
    // with trailing bytes are both rejected rather than silently accepted.
    uint64_t values = 0;
    uint64_t payload = 0;
    if (!checked_mul(rows, cols, &values) || !checked_mul(values, 4, &payload)) {
        return io_fail(err, "'" + path + "': header dimensions " + std::to_string(rows) + "x" +
                                std::to_string(cols) + " overflow the payload size");
    }
    const uint64_t expected = kKmdHeaderBytes + payload;
    if (file_size != expected) {
        return io_fail(err, "'" + path + "': size is " + std::to_string(file_size) +
                                " B but the header implies " + std::to_string(expected) + " B (" +
                                std::to_string(rows) + "x" + std::to_string(cols) + " binary32)");
    }

    out->rows = rows;
    out->cols = cols;
    out->data.assign(static_cast<size_t>(values), 0.0f);
    if (values != 0 && !read_exact(&in, out->data.data(), static_cast<size_t>(payload))) {
        return io_fail(err, "'" + path + "': payload could not be read");
    }
    return kExitOk;
}

int read_kma(const std::string &path, std::vector<int32_t> *out, std::string *err)
{
    if (!host_is_little_endian()) {
        return io_fail(err, "host is big-endian; BLD-004 fixes little-endian byte order on disk");
    }

    std::ifstream in;
    uint64_t file_size = 0;
    if (!open_sized(path, &in, &file_size)) {
        return io_fail(err, "cannot open '" + path + "' for reading");
    }
    if (file_size < kKmaHeaderBytes) {
        return io_fail(err, "'" + path + "' is shorter than the .kma header");
    }

    char magic[4] = {0, 0, 0, 0};
    uint32_t version = 0;
    uint64_t count = 0;
    if (!read_exact(&in, magic, 4) || !read_exact(&in, &version, 4) ||
        !read_exact(&in, &count, 8)) {
        return io_fail(err, "'" + path + "': header could not be read");
    }
    if (std::memcmp(magic, "KMA1", 4) != 0) {
        return io_fail(err, "'" + path + "': bad magic, expected KMA1");
    }
    if (version != kFormatVersion) {
        return io_fail(err, "'" + path + "': unsupported .kma version " + std::to_string(version));
    }

    uint64_t payload = 0;
    if (!checked_mul(count, 4, &payload)) {
        return io_fail(err, "'" + path + "': N=" + std::to_string(count) + " overflows the payload");
    }
    const uint64_t expected = kKmaHeaderBytes + payload;
    if (file_size != expected) {
        return io_fail(err, "'" + path + "': size is " + std::to_string(file_size) +
                                " B but the header implies " + std::to_string(expected) + " B");
    }

    out->assign(static_cast<size_t>(count), 0);
    if (count != 0 && !read_exact(&in, out->data(), static_cast<size_t>(payload))) {
        return io_fail(err, "'" + path + "': payload could not be read");
    }
    return kExitOk;
}

int read_kmd_expect(const std::string &path, uint64_t rows, uint64_t cols, Matrix *out,
                    std::string *err)
{
    const int code = read_kmd(path, out, err);
    if (code != kExitOk) {
        return code;
    }
    if (out->rows != rows || out->cols != cols) {
        if (err != nullptr) {
            *err = "'" + path + "' holds " + std::to_string(out->rows) + "x" +
                   std::to_string(out->cols) + " values but " + std::to_string(rows) + "x" +
                   std::to_string(cols) + " was declared";
        }
        return kExitInvalidInput;  // shape mismatch is a validation failure (FR-002)
    }
    return kExitOk;
}

int write_kmd(const std::string &path, uint64_t rows, uint64_t cols, const float *data,
              std::string *err)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return io_fail(err, "cannot open '" + path + "' for writing");
    }
    const uint32_t version = kFormatVersion;
    out.write("KMD1", 4);
    out.write(reinterpret_cast<const char *>(&version), 4);
    out.write(reinterpret_cast<const char *>(&rows), 8);
    out.write(reinterpret_cast<const char *>(&cols), 8);

    uint64_t values = 0;
    if (!checked_mul(rows, cols, &values)) {
        return io_fail(err, "'" + path + "': dimensions overflow the payload size");
    }
    if (values != 0) {
        out.write(reinterpret_cast<const char *>(data),
                  static_cast<std::streamsize>(values * sizeof(float)));
    }
    out.flush();
    if (!out.good()) {
        return io_fail(err, "'" + path + "': write failed");
    }
    return kExitOk;
}

int write_kma(const std::string &path, const std::vector<int32_t> &assignments, std::string *err)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return io_fail(err, "cannot open '" + path + "' for writing");
    }
    const uint32_t version = kFormatVersion;
    const uint64_t count = assignments.size();
    out.write("KMA1", 4);
    out.write(reinterpret_cast<const char *>(&version), 4);
    out.write(reinterpret_cast<const char *>(&count), 8);
    if (count != 0) {
        out.write(reinterpret_cast<const char *>(assignments.data()),
                  static_cast<std::streamsize>(count * sizeof(int32_t)));
    }
    out.flush();
    if (!out.good()) {
        return io_fail(err, "'" + path + "': write failed");
    }
    return kExitOk;
}

}  // namespace kmeans
