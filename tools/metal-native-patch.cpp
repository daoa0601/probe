#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

void print_usage(const char * program) {
    std::fprintf(
        stderr,
        "usage: %s INPUT OUTPUT BYTE_OFFSET EXPECTED_BYTE REPLACEMENT_BYTE\n"
        "       integer arguments accept decimal or 0x-prefixed hexadecimal\n",
        program);
}

bool parse_uint64(const char * text, const char * name, uint64_t * value) {
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') {
        std::fprintf(stderr, "invalid %s: %s\n", name, text);
        return false;
    }
    *value = static_cast<uint64_t>(parsed);
    return true;
}

}  // namespace

int main(int argc, const char * argv[]) {
    if (argc != 6) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (std::string(argv[1]) == argv[2]) {
        std::fprintf(stderr, "input and output paths must differ\n");
        return EXIT_FAILURE;
    }

    uint64_t offset = 0;
    uint64_t expected = 0;
    uint64_t replacement = 0;
    if (!parse_uint64(argv[3], "byte offset", &offset) ||
        !parse_uint64(argv[4], "expected byte", &expected) ||
        !parse_uint64(argv[5], "replacement byte", &replacement) ||
        expected > UINT8_MAX || replacement > UINT8_MAX) {
        return EXIT_FAILURE;
    }

    std::ifstream input(argv[1], std::ios::binary);
    if (!input) {
        std::perror("open input");
        return EXIT_FAILURE;
    }
    std::vector<uint8_t> contents{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    if (offset >= contents.size()) {
        std::fprintf(
            stderr,
            "byte offset %llu is outside %zu-byte input\n",
            static_cast<unsigned long long>(offset),
            contents.size());
        return EXIT_FAILURE;
    }
    if (contents[offset] != expected) {
        std::fprintf(
            stderr,
            "byte mismatch at offset %llu: expected 0x%02llx, found 0x%02x\n",
            static_cast<unsigned long long>(offset),
            static_cast<unsigned long long>(expected),
            contents[offset]);
        return EXIT_FAILURE;
    }

    contents[offset] = static_cast<uint8_t>(replacement);
    std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
    if (!output) {
        std::perror("open output");
        return EXIT_FAILURE;
    }
    output.write(
        reinterpret_cast<const char *>(contents.data()),
        static_cast<std::streamsize>(contents.size()));
    output.close();
    if (!output) {
        std::perror("write output");
        return EXIT_FAILURE;
    }

    std::printf(
        "{"
        "\"input\":\"%s\","
        "\"output\":\"%s\","
        "\"size\":%zu,"
        "\"byte_offset\":%llu,"
        "\"before\":%llu,"
        "\"after\":%llu"
        "}\n",
        argv[1],
        argv[2],
        contents.size(),
        static_cast<unsigned long long>(offset),
        static_cast<unsigned long long>(expected),
        static_cast<unsigned long long>(replacement));
    return EXIT_SUCCESS;
}
