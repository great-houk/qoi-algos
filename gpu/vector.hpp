#pragma once
#include <cstddef>

template <typename T, std::size_t MaxCapacity>
class Vector {
public:
    using value_type     = T;
    using size_type      = std::size_t;
    using reference      = T &;
    using const_reference = const T &;

    __host__ __device__
    Vector() : size_(0) {}

    // ----- element access -----

    __host__ __device__
    reference operator[](size_type pos) {
        return data_[pos];
    }

    __host__ __device__
    const_reference operator[](size_type pos) const {
        return data_[pos];
    }

    // ----- capacity -----

    __host__ __device__
    size_type size() const {
        return size_;
    }


    // ----- modifiers -----

    __host__ __device__
    void push_back(const T &value) {
        // IMPORTANT: check bounds in debug; in release you might drop this
        if (size_ < MaxCapacity) {
            data_[size_++] = value;
        }
        // else: overflow; you can add assert or custom handling if you want
    }

private:
    T          data_[MaxCapacity];
    size_type  size_;
};
